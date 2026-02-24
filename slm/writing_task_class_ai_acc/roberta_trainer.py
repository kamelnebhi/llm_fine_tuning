import os
os.environ["WANDB_DISABLED"] = "true"

import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    classification_report,
)
from scipy.stats import pearsonr
import boto3

# ── constants ──────────────────────────────────────────────────────────────────
region_name    = 'us-east-1'
profile_name   = 'lsd-sandbox'
bucket_name    = 'sagemaker-studio-oxs6vznjds'
CSV_PATH       = "data_v2/sample_data_acc_gec.csv"
FIXED_EVAL_DIR = "data_v2/fixed_eval"   # val + test saved here once
NUM_LABELS     = 6
LABEL_NAMES    = [f"Score {i}" for i in range(NUM_LABELS)]

median_map = {
    "A1": 2, "A2": 5, "B1": 8, "B2": 11, "C1": 14, "C2": 16
}

session   = boto3.Session()
s3_client = session.client('s3')


# ============================================================
# ORDINAL TRAINER  (unchanged)
# ============================================================
class OrdinalTrainer(Trainer):
    def __init__(self, *args, sigma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        num_labels = logits.shape[-1]
        device     = logits.device
        classes    = torch.arange(num_labels, device=device).float()

        soft_labels = torch.exp(
            -((classes.unsqueeze(0) - labels.unsqueeze(1).float()) ** 2)
            / (2 * self.sigma ** 2)
        )
        soft_labels = soft_labels / soft_labels.sum(dim=-1, keepdim=True)

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


# ============================================================
# SHARED TEXT PREPROCESSING
# ============================================================
def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Text engineering shared across all splits.
    df_raw must already have an 'orig_idx' column (set before any row filtering
    so indices are stable across different agreement-threshold runs).
    Returns df with: orig_idx, text, task_id, ef_level, label, agreement_percentage
    """
    df = df_raw.copy()
    df['ef_level'] = df.apply(
        lambda row: median_map[row['cefr_level']] if pd.isna(row['ef_level'])
        else row['ef_level'],
        axis=1,
    )
    df['text'] = (
        "Prompt Level: " + df['ef_level'].astype(str)
        + " Prompt: "     + df['activity_instructions']
        + " Response: "   + df['student_submission']
        + " Correction: " + df['correction']
    )
    keep = ["orig_idx", "text", "task_id", "ef_level",
            "majority_value", "agreement_percentage"]
    df = df[keep].rename(columns={'majority_value': 'label'})
    df.dropna(subset=['label', 'text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _cast_labels(ds: Dataset) -> Dataset:
    """Cast 'label' column to ClassLabel so stratified splits work."""
    feats = ds.features.copy()
    feats["label"] = ClassLabel(names=[str(i) for i in range(NUM_LABELS)])
    return ds.cast(feats)



def load_jsonl_as_dataset(path: str) -> Dataset:
    """Remplacement de Dataset.from_json() qui bug sur filesystem local."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_pandas(pd.DataFrame(records))

# ============================================================
# FIXED EVAL SETS  ← built ONCE from agreement = 100 %
# ============================================================
def build_fixed_eval_sets(csv_path: str, tokenizer, force_rebuild: bool = False):
    os.makedirs(FIXED_EVAL_DIR, exist_ok=True)
    meta_path = os.path.join(FIXED_EVAL_DIR, "eval_meta.json")
    val_path  = os.path.join(FIXED_EVAL_DIR, "validation.jsonl")
    test_path = os.path.join(FIXED_EVAL_DIR, "test.jsonl")

    if not force_rebuild and os.path.exists(meta_path):
        print("↩  Loading existing fixed eval sets from disk …")
        with open(meta_path) as f:
            meta = json.load(f)
        val_ds  = _cast_labels(load_jsonl_as_dataset(val_path))
        test_ds = _cast_labels(load_jsonl_as_dataset(test_path))
        print(f"   val={len(val_ds)}, test={len(test_ds)}, "
              f"max_length={meta['max_length']}")
        return val_ds, test_ds, set(meta["eval_orig_indices"]), meta["max_length"]

    print("Building fixed eval sets …")
    df_raw = pd.read_csv(csv_path)
    df_raw['orig_idx'] = df_raw.index

    df = preprocess_df(df_raw)
    df["agreement_percentage"] = pd.to_numeric(
        df["agreement_percentage"], errors='coerce'
    )

    # ── max_length depuis le pool le plus large ────────────────────────────
    print("  Computing max_length from agreement>=60% pool …")
    df_pool = df[df["agreement_percentage"] >= 60]
    lengths  = tokenizer(
        df_pool['text'].tolist(),
        truncation=False, padding=False, return_length=True
    )["length"]
    p95        = int(np.percentile(lengths, 95))
    max_length = min(max(p95, 128), 512)
    print(f"  token length — median={int(np.median(lengths))}, "
          f"95th={p95}, chosen max_length={max_length}")

    # ── isoler les deux pools ──────────────────────────────────────────────
    df_100   = df[df["agreement_percentage"] == 100].copy()
    df_60_80 = df[
        (df["agreement_percentage"] >= 60) &
        (df["agreement_percentage"] <= 80)
    ].copy()

    print(f"\n  Pool agreement=100%    : {len(df_100)} samples")
    print(f"  Pool agreement 60-80%  : {len(df_60_80)} samples")

    # ── tirage stratifié depuis chaque pool ───────────────────────────────
    # Objectif : val ~N samples  (60% de 100%, 40% de 60-80%)
    # On prend 15% de chaque pool puis on compose le mix
    EVAL_RATIO = 0.15   # part de chaque pool réservée à val+test

    def stratified_sample(df_in: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
        """Tirage stratifié par label."""
        return (
            df_in.groupby('label', group_keys=False)
                 .apply(lambda x: x.sample(
                     max(1, int(len(x) * ratio)),
                     random_state=seed
                 ))
        )

    sample_100   = stratified_sample(df_100,   EVAL_RATIO, seed=42)
    sample_60_80 = stratified_sample(df_60_80, EVAL_RATIO, seed=42)

    print(f"\n  Tirage éval depuis 100%    : {len(sample_100)} samples")
    print(f"  Tirage éval depuis 60-80%  : {len(sample_60_80)} samples")

    # ── composition du mix 60% / 40% ──────────────────────────────────────
    n_total    = len(sample_100) + len(sample_60_80)
    n_from_100 = int(n_total * 0.60)
    n_from_noisy = n_total - n_from_100

    # Ajuste si pas assez de samples dans un pool
    n_from_100   = min(n_from_100,   len(sample_100))
    n_from_noisy = min(n_from_noisy, len(sample_60_80))

    final_100   = sample_100.sample(n=n_from_100,   random_state=42)
    final_noisy = sample_60_80.sample(n=n_from_noisy, random_state=42)

    df_eval = pd.concat([final_100, final_noisy], ignore_index=True)
    df_eval["is_100_agreement"] = df_eval["orig_idx"].isin(final_100["orig_idx"])

    print(f"\n  Eval pool final : {len(df_eval)} samples")
    print(f"    → {n_from_100} samples agreement=100%  "
          f"({n_from_100/len(df_eval)*100:.1f}%)")
    print(f"    → {n_from_noisy} samples agreement=60-80%  "
          f"({n_from_noisy/len(df_eval)*100:.1f}%)")
    print("\n  Label distribution eval pool:")
    print(df_eval['label'].value_counts().sort_index())

    # ── split val / test (50/50 stratifié) ────────────────────────────────
    ds_eval = _cast_labels(
        Dataset.from_pandas(
            df_eval[["orig_idx", "text", "task_id", "ef_level",
                      "label", "is_100_agreement"]],
            preserve_index=False,
        )
    )

    split   = ds_eval.train_test_split(
        test_size=0.50, seed=42, stratify_by_column="label"
    )
    val_ds  = split["train"]
    test_ds = split["test"]

    eval_orig_indices = set(val_ds["orig_idx"]) | set(test_ds["orig_idx"])

    print(f"\n  val  : {len(val_ds)} samples "
          f"(100%: {sum(val_ds['is_100_agreement'])}, "
          f"noisy: {sum(not x for x in val_ds['is_100_agreement'])})")
    print(f"  test : {len(test_ds)} samples "
          f"(100%: {sum(test_ds['is_100_agreement'])}, "
          f"noisy: {sum(not x for x in test_ds['is_100_agreement'])})")

    # persist
    save_split_to_jsonl(val_ds,  val_path)
    save_split_to_jsonl(test_ds, test_path)

    meta = {
        "eval_orig_indices": list(eval_orig_indices),
        "max_length":        max_length,
        "n_val":             len(val_ds),
        "n_test":            len(test_ds),
        "mix_100_pct":       60,
        "mix_noisy_pct":     40,
        "agreement_noisy_range": "60-80",
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved → {FIXED_EVAL_DIR}/")
    return val_ds, test_ds, eval_orig_indices, max_length

# ============================================================
# TRAINING SET BUILDER  (one per experiment)
# ============================================================
def prepare_train_set(
    csv_path: str,
    eval_orig_indices: set,
    agreement_threshold: int,
    experiment_dir: str,
) -> Dataset:
    df_raw = pd.read_csv(csv_path)
    df_raw['orig_idx'] = df_raw.index

    print(f"\n[DEBUG] df_raw shape: {df_raw.shape}")
    print(f"[DEBUG] columns: {df_raw.columns.tolist()}")
    print(f"[DEBUG] agreement_percentage dtype: {df_raw['agreement_percentage'].dtype}")
    print(f"[DEBUG] agreement_percentage sample values: {df_raw['agreement_percentage'].dropna().unique()[:10]}")
    print(f"[DEBUG] agreement_percentage NaN count: {df_raw['agreement_percentage'].isna().sum()}")

    df = preprocess_df(df_raw)
    print(f"[DEBUG] after preprocess_df shape: {df.shape}")

    # ── filtre par threshold ──────────────────────────────────────────────
    # cast explicite pour éviter les bugs de dtype (string vs float)
    df["agreement_percentage"] = pd.to_numeric(
        df["agreement_percentage"], errors='coerce'
    )
    df_thresh = df[df["agreement_percentage"] >= agreement_threshold]
    print(f"[DEBUG] after agreement>={agreement_threshold}% filter: {len(df_thresh)} rows")

    # ── exclusion des eval rows ───────────────────────────────────────────
    print(f"[DEBUG] eval_orig_indices size: {len(eval_orig_indices)}")
    print(f"[DEBUG] sample eval_orig_indices: {list(eval_orig_indices)[:5]}")
    print(f"[DEBUG] sample df orig_idx: {df_thresh['orig_idx'].head().tolist()}")

    df_train = df_thresh[~df_thresh['orig_idx'].isin(eval_orig_indices)].copy()
    print(f"[DEBUG] after eval exclusion: {len(df_train)} rows")

    df_train.reset_index(drop=True, inplace=True)

    print(f"\n[Train | agreement>={agreement_threshold}%]  n_samples={len(df_train)}")
    print("Label distribution:")
    print(df_train['label'].value_counts().sort_index())

    if len(df_train) == 0:
        raise ValueError(
            f"Train set is empty for threshold={agreement_threshold}%. "
            f"Vérifier les colonnes et valeurs de agreement_percentage."
        )

    ds = _cast_labels(
        Dataset.from_pandas(
            df_train[["orig_idx", "text", "task_id", "ef_level", "label"]],
            preserve_index=False,
        )
    )

    os.makedirs(experiment_dir, exist_ok=True)
    save_split_to_jsonl(ds, os.path.join(experiment_dir, "train.jsonl"))
    return ds

# ============================================================
# HELPERS
# ============================================================
def save_split_to_jsonl(dataset_split, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for record in dataset_split:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def tokenize_ds(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    return ds.map(
        lambda ex: tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


# ============================================================
# METRICS  (unchanged)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy":              float('nan'),
        "precision":             float('nan'),
        "recall":                float('nan'),
        "f1":                    float('nan'),
        "cohen_kappa":           float('nan'),
        "pearson_corr":          float('nan'),
        "classification_report": {},
    }

    try:
        if len(labels) >= 2 and len(set(labels)) > 1 and len(set(predictions)) > 1:
            metrics["accuracy"] = accuracy_score(labels, predictions)
            metrics["precision"], metrics["recall"], metrics["f1"], _ = (
                precision_recall_fscore_support(
                    labels, predictions, average="weighted", zero_division=0
                )
            )
            metrics["cohen_kappa"] = cohen_kappa_score(
                labels, predictions, weights="quadratic"
            )
            metrics["pearson_corr"], _ = pearsonr(labels, predictions)
            metrics["classification_report"] = classification_report(
                labels, predictions, output_dict=True, zero_division=0
            )
    except Exception as e:
        print(f"[!] Error in compute_metrics: {e}")

    return metrics


def safe_metrics(ref_labels, predicted_labels):
    result = {
        "accuracy": float('nan'), "precision": float('nan'),
        "recall":   float('nan'), "f1":        float('nan'),
        "ck":       float('nan'), "pearson":   float('nan'),
    }
    if (len(ref_labels) < 2
            or len(set(ref_labels)) <= 1
            or len(set(predicted_labels)) <= 1):
        return result
    try:
        result["ck"] = round(
            cohen_kappa_score(predicted_labels, ref_labels, weights="quadratic"), 4
        )
    except Exception as e:
        print(f"[!] Cohen Kappa: {e}")
    try:
        result["pearson"], _ = pearsonr(ref_labels, predicted_labels)
        result["pearson"] = round(result["pearson"], 4)
    except Exception as e:
        print(f"[!] Pearson: {e}")
    try:
        result["accuracy"] = round(accuracy_score(ref_labels, predicted_labels), 4)
        p, r, f, _ = precision_recall_fscore_support(
            ref_labels, predicted_labels, average="weighted", zero_division=0
        )
        result["precision"] = round(p, 4)
        result["recall"]    = round(r, 4)
        result["f1"]        = round(f, 4)
    except Exception as e:
        print(f"[!] Other metrics: {e}")
    return result


# ============================================================
# TRAINING  (unchanged)
# ============================================================
def train_model(tokenized_train, tokenized_test, num_labels, output_dir, tokenizer):
    model = RobertaForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-large",
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="cohen_kappa",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = OrdinalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,   # fixed test set used for early stopping
        compute_metrics=compute_metrics,
        sigma=1.0,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    best_model_path = trainer.state.best_model_checkpoint
    print(f"Best model checkpoint: {best_model_path}")

    trainer.save_model(best_model_path)
    model.save_pretrained(best_model_path, from_pt=True)
    tokenizer.save_pretrained(best_model_path)

    return trainer, eval_results


# ============================================================
# DETAILED EVALUATION  — now scoped per experiment
# ============================================================
def detailed_evaluation(trainer, tokenized_valid, experiment_name: str = "experiment"):
    """
    Identical analysis as before but:
      - output files written to  results/{experiment_name}/
      - returns a summary dict for side-by-side comparison
    """
    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    unique_tasks  = set(tokenized_valid["task_id"])
    unique_levels = set(tokenized_valid["ef_level"])

    # ── 1. GLOBAL ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"[{experiment_name}]  GLOBAL EVALUATION ON VALIDATION SET")
    print("=" * 80)

    global_preds     = trainer.predict(tokenized_valid)
    global_predicted = np.argmax(global_preds.predictions, axis=-1)
    global_refs      = global_preds.label_ids

    print("\n📊 Classification Report:")
    print(classification_report(
        global_refs, global_predicted,
        target_names=LABEL_NAMES, zero_division=0
    ))

    global_acc     = accuracy_score(global_refs, global_predicted)
    global_qwk     = cohen_kappa_score(global_refs, global_predicted, weights="quadratic")
    global_lwk     = cohen_kappa_score(global_refs, global_predicted, weights="linear")
    global_pearson, _ = pearsonr(global_refs, global_predicted)
    print(f"✅ Accuracy:        {global_acc:.4f}")
    print(f"✅ QWK (quadratic): {global_qwk:.4f}")
    print(f"✅ LWK (linear):    {global_lwk:.4f}")
    print(f"✅ Pearson:         {global_pearson:.4f}")


    # ── Métrique sur agreement=100% uniquement ────────────────────────────────
    if "is_100_agreement" in tokenized_valid.column_names:
        mask_100 = np.array(tokenized_valid["is_100_agreement"])
        if mask_100.sum() >= 2:
            refs_100  = global_refs[mask_100]
            preds_100 = global_predicted[mask_100]
            qwk_100   = cohen_kappa_score(refs_100, preds_100, weights="quadratic")
            acc_100   = accuracy_score(refs_100, preds_100)
            print(f"\n📊 Sur agreement=100% uniquement (n={mask_100.sum()}):")
            print(f"   QWK      : {qwk_100:.4f}")
            print(f"   Accuracy : {acc_100:.4f}")

    # ── 2. CONFUSION MATRIX ───────────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    cm     = confusion_matrix(global_refs, global_predicted)
    cm_df  = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df.to_string())
    cm_df.to_csv(f"{out_dir}/confusion_matrix.csv")

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm_df = pd.DataFrame(
        np.round(cm_norm, 3), index=LABEL_NAMES, columns=LABEL_NAMES
    )
    print("\n📊 Normalized Confusion Matrix (recall per class):")
    print(cm_norm_df.to_string())
    cm_norm_df.to_csv(f"{out_dir}/confusion_matrix_normalized.csv")

    # ── 3. ERROR ANALYSIS ─────────────────────────────────────────────────────
    errors          = global_predicted != global_refs
    error_distances = global_predicted[errors].astype(int) - global_refs[errors].astype(int)
    print(f"\n🔴 Total errors: {errors.sum()} / {len(global_refs)} "
          f"({errors.sum() / len(global_refs) * 100:.1f}%)")
    dist_counts = Counter(error_distances)
    print("\n📊 Error distance distribution (predicted − true):")
    for dist in sorted(dist_counts):
        direction = "↑ over" if dist > 0 else "↓ under"
        print(f"  Distance {dist:+d} ({direction}): {dist_counts[dist]}")

    global_adj_acc = (
        np.abs(global_predicted.astype(int) - global_refs.astype(int)) <= 1
    ).mean()
    global_mae = np.abs(
        global_predicted.astype(int) - global_refs.astype(int)
    ).mean()
    print(f"\n✅ Adjacent accuracy (±1): {global_adj_acc:.4f}")
    print(f"✅ MAE:                    {global_mae:.4f}")

    # ── 4. PER-CLASS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-CLASS ANALYSIS")
    print("=" * 80)
    per_class_results = []
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        mask = global_refs == cls_idx
        if mask.sum() == 0:
            continue
        cls_preds  = global_predicted[mask]
        cls_refs   = global_refs[mask]
        n_total    = len(cls_refs)
        cls_acc    = (cls_preds == cls_refs).sum() / n_total
        cls_adj    = (np.abs(cls_preds.astype(int) - cls_refs.astype(int)) <= 1).mean()
        cls_mae    = np.abs(cls_preds.astype(int)  - cls_refs.astype(int)).mean()
        wrong_mask = cls_preds != cls_refs
        top_conf_str = (
            ", ".join([
                f"{LABEL_NAMES[p]}({c})"
                for p, c in Counter(cls_preds[wrong_mask]).most_common(3)
            ])
            if wrong_mask.sum() > 0 else "None"
        )
        per_class_results.append({
            "class": cls_name, "n_samples": n_total,
            "accuracy": round(cls_acc, 4), "adjacent_acc": round(cls_adj, 4),
            "mae": round(cls_mae, 4), "n_errors": int(wrong_mask.sum()),
            "top_confusions": top_conf_str,
        })
        print(f"  [{cls_name}] n={n_total} | acc={cls_acc:.3f} | "
              f"adj={cls_adj:.3f} | mae={cls_mae:.3f} | confused: {top_conf_str}")
    pd.DataFrame(per_class_results).to_csv(f"{out_dir}/per_class_analysis.csv", index=False)

    # ── 5. BY EF_LEVEL ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION BY EF_LEVEL")
    print("=" * 80)
    results_levels = []
    for lv in sorted(unique_levels):
        sub_ds = tokenized_valid.filter(lambda ex: ex['ef_level'] == lv)
        if len(sub_ds) < 2:
            continue
        preds       = trainer.predict(sub_ds)
        pred_labels = np.argmax(preds.predictions, axis=-1)
        ref_labels  = preds.label_ids
        m    = safe_metrics(ref_labels, pred_labels)
        adj  = (np.abs(pred_labels.astype(int) - ref_labels.astype(int)) <= 1).mean()
        mae_ = np.abs(pred_labels.astype(int)  - ref_labels.astype(int)).mean()
        results_levels.append({
            "ef_level": lv, **m,
            "adjacent_acc": round(adj, 4),
            "mae": round(mae_, 4),
            "n_samples": len(sub_ds),
        })
        print(f"  Level {lv:>2} | n={len(sub_ds):>4} | acc={m['accuracy']:.3f} | "
              f"qwk={m['ck']:.3f} | adj={adj:.3f} | mae={mae_:.3f}")
    pd.DataFrame(results_levels).to_csv(f"{out_dir}/eval_by_level.csv", index=False)

    # ── 6. BY TASK_ID ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION BY TASK_ID")
    print("=" * 80)
    results_tasks = []
    for t in unique_tasks:
        sub_ds = tokenized_valid.filter(lambda ex: ex['task_id'] == t)
        if len(sub_ds) < 2:
            continue
        preds       = trainer.predict(sub_ds)
        pred_labels = np.argmax(preds.predictions, axis=-1)
        ref_labels  = preds.label_ids
        m    = safe_metrics(ref_labels, pred_labels)
        adj  = (np.abs(pred_labels.astype(int) - ref_labels.astype(int)) <= 1).mean()
        mae_ = np.abs(pred_labels.astype(int)  - ref_labels.astype(int)).mean()
        results_tasks.append({
            "task_id": t, "ef_level": sub_ds["ef_level"][0], **m,
            "adjacent_acc": round(adj, 4), "mae": round(mae_, 4),
            "n_samples": len(sub_ds),
        })
    df_tasks = pd.DataFrame(results_tasks).sort_values("ck", ascending=True)
    df_tasks.to_csv(f"{out_dir}/eval_by_task.csv", index=False)
    print("\n🔴 5 WORST tasks (by QWK):")
    print(df_tasks.head(5)[
        ["task_id", "ef_level", "ck", "accuracy", "mae", "n_samples"]
    ].to_string(index=False))
    print("\n🟢 5 BEST tasks (by QWK):")
    print(df_tasks.tail(5)[
        ["task_id", "ef_level", "ck", "accuracy", "mae", "n_samples"]
    ].to_string(index=False))

    # ── 7. SAVE PREDICTIONS ───────────────────────────────────────────────────
    softmax_probs = torch.softmax(
        torch.tensor(global_preds.predictions), dim=-1
    ).numpy()
    val_df = pd.DataFrame({
        "text":            tokenized_valid["text"],
        "task_id":         tokenized_valid["task_id"],
        "ef_level":        tokenized_valid["ef_level"],
        "true_label":      global_refs,
        "predicted_label": global_predicted,
        "error_distance":  global_predicted.astype(int) - global_refs.astype(int),
        "is_correct":      global_predicted == global_refs,
        "confidence":      softmax_probs.max(axis=1),
        "predicted_probs": [
            {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(row)}
            for row in softmax_probs
        ],
    })
    val_df.to_csv(f"{out_dir}/validation_predictions_full.csv", index=False)
    errors_df = val_df[~val_df["is_correct"]].sort_values(
        "confidence", ascending=False
    )
    errors_df.to_csv(f"{out_dir}/validation_errors_by_confidence.csv", index=False)

    # ── 8. SUMMARY ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SUMMARY — {experiment_name}")
    print("=" * 80)
    print(f"  Accuracy:        {global_acc:.4f}")
    print(f"  QWK:             {global_qwk:.4f}")
    print(f"  LWK:             {global_lwk:.4f}")
    print(f"  Pearson:         {global_pearson:.4f}")
    print(f"  Adjacent Acc:    {global_adj_acc:.4f}")
    print(f"  MAE:             {global_mae:.4f}")
    print(f"  Total samples:   {len(global_refs)}")
    print(f"  Total errors:    {errors.sum()}")
    print(f"  High-conf errs:  {len(errors_df[errors_df['confidence'] > 0.8])}")
    print(f"  Output dir:      {out_dir}/")
    print("=" * 80)

    return {
        "experiment":   experiment_name,
        "accuracy":     round(global_acc,    4),
        "qwk":          round(global_qwk,    4),
        "lwk":          round(global_lwk,    4),
        "pearson":      round(global_pearson, 4),
        "adjacent_acc": round(global_adj_acc, 4),
        "mae":          round(float(global_mae), 4),
        "n_samples":    len(global_refs),
        "n_errors":     int(errors.sum()),
    }


# ============================================================
# S3 UPLOAD  (unchanged)
# ============================================================
def upload_model_to_s3(local_dir, bucket_name, s3_prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative   = os.path.relpath(local_path, local_dir)
            s3_path    = os.path.join(s3_prefix, relative).replace("\\", "/")
            print(f"Uploading {local_path} → s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)


# ============================================================
# MAIN
# ============================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    # ── Step 1 : build / reload fixed eval sets ONCE ──────────────────────────
    # val  → detailed evaluation   (agreement = 100%, never seen during training)
    # test → early-stopping signal during training (agreement = 100%)
    val_ds, test_ds, eval_orig_indices, max_length = build_fixed_eval_sets(
        CSV_PATH, tokenizer, force_rebuild=True
    )

    # ── Step 2 : tokenize fixed eval sets ONCE ────────────────────────────────
    print(f"\nTokenizing fixed eval sets (max_length={max_length}) …")
    tokenized_test  = tokenize_ds(test_ds,  tokenizer, max_length)
    tokenized_valid = tokenize_ds(val_ds,   tokenizer, max_length)


    AGREEMENT_THRESHOLD = 100   # 60  ou  100

    # ── Step 3 : run experiments ───────────────────────────────────────────────
    experiments = [
    {"agreement_threshold": AGREEMENT_THRESHOLD,
     "name": f"agreement_{'gte60' if AGREEMENT_THRESHOLD == 60 else 'eq100'}"},
    ]

    os.makedirs("results", exist_ok=True)
    comparison = []

    for exp in experiments:
        threshold = exp["agreement_threshold"]
        exp_name  = exp["name"]
        output_dir   = f"model_saved/roberta-large-ft-{exp_name}"
        exp_data_dir = f"data_v2/{exp_name}"

        print(f"\n{'='*80}")
        print(f"EXPERIMENT : {exp_name}  (agreement >= {threshold}%)")
        print('='*80)

        # build train — eval rows are excluded regardless of threshold
        train_ds = prepare_train_set(
            CSV_PATH, eval_orig_indices, threshold, exp_data_dir
        )

        print(f"Tokenizing train set …")
        tokenized_train = tokenize_ds(train_ds, tokenizer, max_length)

        # train — both experiments evaluate on the SAME fixed test set
        trainer, eval_results = train_model(
            tokenized_train, tokenized_test, NUM_LABELS, output_dir, tokenizer
        )
        print(f"\n[{exp_name}] Test-set eval results (used for early stopping):")
        print(eval_results)

        # detailed evaluation — SAME fixed validation set for both
        summary = detailed_evaluation(
            trainer, tokenized_valid, experiment_name=exp_name
        )
        comparison.append(summary)

    # ── Step 4 : side-by-side comparison ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON  (identical val set for both experiments)")
    print("=" * 80)
    df_cmp = pd.DataFrame(comparison).set_index("experiment")
    print(df_cmp.T.to_string())          # metrics as rows, experiments as cols
    df_cmp.to_csv("results/comparison_summary.csv")
    print("\nSaved → results/comparison_summary.csv")


if __name__ == "__main__":
    main()
