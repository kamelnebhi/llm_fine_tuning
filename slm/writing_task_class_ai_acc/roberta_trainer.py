import os
os.environ["WANDB_DISABLED"] = "true"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    RobertaModel,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
FIXED_EVAL_DIR = "data_v2/fixed_eval"
NUM_LABELS     = 6
LABEL_NAMES    = [f"Score {i}" for i in range(NUM_LABELS)]

median_map = {
    "A1": 2, "A2": 5, "B1": 8, "B2": 11, "C1": 14, "C2": 16
}

session   = boto3.Session()
s3_client = session.client('s3')


# ============================================================
# TWO-TOWER MODEL  (shared RoBERTa-large encoder)
# ============================================================
class TwoTowerRoberta(nn.Module):
    """
    Shared-encoder two-tower model.
    Tower A : level + prompt + student response  (up to max_length tokens)
    Tower B : correction                          (up to max_length tokens)
    Head    : [CLS_A ; CLS_B ; CLS_A - CLS_B]  →  3 * 1024 → 512 → num_labels
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size   # 1024 for roberta-large
        self.num_labels = num_labels

        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def encode(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]   # [CLS] token

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        labels=None,
        sample_weight=None,
        **kwargs,
    ):
        cls_a    = self.encode(input_ids_a, attention_mask_a)
        cls_b    = self.encode(input_ids_b, attention_mask_b)
        # Enriched fusion: gives the model an explicit difference signal
        combined = torch.cat([cls_a, cls_b, cls_a - cls_b], dim=-1)
        logits   = self.classifier(combined)

        return SequenceClassifierOutput(loss=None, logits=logits)


# ============================================================
# DATA COLLATOR  — handles two-tower fields + optional sample_weight
# ============================================================
class TwoTowerDataCollator:
    """
    Collates only tensor fields.
    - Renames 'label' → 'labels' as expected by the Trainer.
    - 'sample_weight' is optional (absent in eval sets).
    """
    TENSOR_FIELDS = [
        "input_ids_a", "attention_mask_a",
        "input_ids_b", "attention_mask_b",
        "label", "sample_weight",
    ]

    def __call__(self, features):
        batch = {}
        for field in self.TENSOR_FIELDS:
            if field not in features[0]:
                continue
            dtype = torch.float32 if field == "sample_weight" else torch.long
            batch[field] = torch.tensor(
                [f[field] for f in features], dtype=dtype
            )
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        return batch


# ============================================================
# ORDINAL TRAINER  — soft Gaussian labels + per-sample weights
# ============================================================
class OrdinalTrainer(Trainer):
    def __init__(self, *args, sigma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)   # absent in eval
        outputs        = model(**inputs)
        logits         = outputs.logits

        num_labels = logits.shape[-1]
        device     = logits.device
        classes    = torch.arange(num_labels, device=device).float()

        # Gaussian soft labels centred on the true class
        soft_labels = torch.exp(
            -((classes.unsqueeze(0) - labels.unsqueeze(1).float()) ** 2)
            / (2 * self.sigma ** 2)
        )
        soft_labels = soft_labels / soft_labels.sum(dim=-1, keepdim=True)

        log_probs       = torch.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_labels * log_probs).sum(dim=-1)

        # Apply combined agreement × class-frequency weight
        if sample_weights is not None:
            loss = (per_sample_loss * sample_weights.to(device)).mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss

    # ── safe checkpoint saving for custom nn.Module ────────────────────────
    def _save_model(self, output_dir, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


# ============================================================
# SHARED TEXT PREPROCESSING  — now produces text_a / text_b
# ============================================================
def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['ef_level'] = df.apply(
        lambda row: median_map[row['cefr_level']] if pd.isna(row['ef_level'])
        else row['ef_level'],
        axis=1,
    )
    # Tower A — context + student response
    df['text_a'] = (
        "Prompt Level: " + df['ef_level'].astype(str)
        + " Prompt: "    + df['activity_instructions'].astype(str)
        + " Response: "  + df['student_submission'].astype(str)
    )
    # Tower B — expert correction only
    df['text_b'] = df['correction'].astype(str)

    keep = ["orig_idx", "text_a", "text_b", "task_id", "ef_level",
            "majority_value", "agreement_percentage"]
    df = df[keep].rename(columns={'majority_value': 'label'})
    df.dropna(subset=['label', 'text_a', 'text_b'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _cast_labels(ds: Dataset) -> Dataset:
    feats          = ds.features.copy()
    feats["label"] = ClassLabel(names=[str(i) for i in range(NUM_LABELS)])
    return ds.cast(feats)


def load_jsonl_as_dataset(path: str) -> Dataset:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_pandas(pd.DataFrame(records))


# ============================================================
# TOKENIZATION  — independent budget per tower
# ============================================================
def tokenize_ds(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    """
    Each tower is tokenised separately up to max_length.
    Tower A uses truncation='only_first' on the combined string
    (level+prompt+response) to never cut the correction.
    """
    def _tokenize(examples):
        enc_a = tokenizer(
            examples["text_a"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc_b = tokenizer(
            examples["text_b"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }
    return ds.map(_tokenize, batched=True)


# ============================================================
# FIXED EVAL SETS  ← built ONCE
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
    df_raw             = pd.read_csv(csv_path)
    df_raw['orig_idx'] = df_raw.index
    df                 = preprocess_df(df_raw)
    df["agreement_percentage"] = pd.to_numeric(
        df["agreement_percentage"], errors='coerce'
    )

    # ── max_length: use the longer of the two towers at p95 ───────────────
    print("  Computing max_length from agreement>=60% pool …")
    df_pool   = df[df["agreement_percentage"] >= 60]
    lengths_a = tokenizer(
        df_pool['text_a'].tolist(),
        truncation=False, padding=False, return_length=True,
    )["length"]
    lengths_b = tokenizer(
        df_pool['text_b'].tolist(),
        truncation=False, padding=False, return_length=True,
    )["length"]
    p95_a      = int(np.percentile(lengths_a, 95))
    p95_b      = int(np.percentile(lengths_b, 95))
    max_length = min(max(max(p95_a, p95_b), 128), 512)
    print(f"  text_a — median={int(np.median(lengths_a))}, 95th={p95_a}")
    print(f"  text_b — median={int(np.median(lengths_b))}, 95th={p95_b}")
    print(f"  chosen max_length={max_length} (applied to each tower independently)")

    df_100   = df[df["agreement_percentage"] == 100].copy()
    df_60_80 = df[
        (df["agreement_percentage"] >= 60) &
        (df["agreement_percentage"] <= 80)
    ].copy()

    print(f"\n  Pool agreement=100%    : {len(df_100)} samples")
    print(f"  Pool agreement 60-80%  : {len(df_60_80)} samples")

    EVAL_RATIO = 0.15

    def stratified_sample(df_in, ratio, seed):
        return (
            df_in.groupby('label', group_keys=False)
                 .apply(lambda x: x.sample(max(1, int(len(x) * ratio)),
                                            random_state=seed))
        )

    sample_100   = stratified_sample(df_100,   EVAL_RATIO, seed=42)
    sample_60_80 = stratified_sample(df_60_80, EVAL_RATIO, seed=42)

    n_total      = len(sample_100) + len(sample_60_80)
    n_from_100   = min(int(n_total * 0.60), len(sample_100))
    n_from_noisy = min(n_total - n_from_100, len(sample_60_80))

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

    ds_eval = _cast_labels(
        Dataset.from_pandas(
            df_eval[["orig_idx", "text_a", "text_b", "task_id", "ef_level",
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

    save_split_to_jsonl(val_ds,  val_path)
    save_split_to_jsonl(test_ds, test_path)

    meta = {
        "eval_orig_indices":     list(eval_orig_indices),
        "max_length":            max_length,
        "n_val":                 len(val_ds),
        "n_test":                len(test_ds),
        "mix_100_pct":           60,
        "mix_noisy_pct":         40,
        "agreement_noisy_range": "60-80",
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved → {FIXED_EVAL_DIR}/")
    return val_ds, test_ds, eval_orig_indices, max_length


# ============================================================
# TRAINING SET BUILDER  — with combined sample weights
# ============================================================
def prepare_train_set(
    csv_path: str,
    eval_orig_indices: set,
    agreement_threshold: int,
    experiment_dir: str,
) -> Dataset:
    df_raw             = pd.read_csv(csv_path)
    df_raw['orig_idx'] = df_raw.index
    df                 = preprocess_df(df_raw)

    df["agreement_percentage"] = pd.to_numeric(
        df["agreement_percentage"], errors='coerce'
    )
    df_thresh = df[df["agreement_percentage"] >= agreement_threshold]
    df_train  = df_thresh[~df_thresh['orig_idx'].isin(eval_orig_indices)].copy()
    df_train.reset_index(drop=True, inplace=True)

    if len(df_train) == 0:
        raise ValueError(f"Train set is empty for threshold={agreement_threshold}%.")

    # ── Agreement weight  (continuous, not binary) ────────────────────────
    df_train["agreement_weight"] = df_train["agreement_percentage"] / 100.0

    # ── Class weight  (inverse frequency) ────────────────────────────────
    label_counts = df_train["label"].value_counts()
    total        = len(df_train)
    cw_map       = {
        cls: total / (NUM_LABELS * cnt)
        for cls, cnt in label_counts.items()
    }
    df_train["class_weight"] = df_train["label"].map(cw_map)

    # ── Combined weight, normalised so mean = 1.0 ─────────────────────────
    df_train["sample_weight"]  = (
        df_train["agreement_weight"] * df_train["class_weight"]
    )
    df_train["sample_weight"] /= df_train["sample_weight"].mean()

    print(f"\n[Train | agreement>={agreement_threshold}%]  n_samples={len(df_train)}")
    print("Label distribution:")
    print(df_train['label'].value_counts().sort_index())
    print(f"Sample weight — min={df_train['sample_weight'].min():.3f}, "
          f"max={df_train['sample_weight'].max():.3f}, "
          f"mean={df_train['sample_weight'].mean():.3f}")

    ds = _cast_labels(
        Dataset.from_pandas(
            df_train[["orig_idx", "text_a", "text_b", "task_id",
                       "ef_level", "label", "sample_weight"]],
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


# ============================================================
# METRICS  (unchanged)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy":     float('nan'),
        "precision":    float('nan'),
        "recall":       float('nan'),
        "f1":           float('nan'),
        "cohen_kappa":  float('nan'),
        "pearson_corr": float('nan'),
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
        result["pearson"]    = round(result["pearson"], 4)
    except Exception as e:
        print(f"[!] Pearson: {e}")
    try:
        result["accuracy"] = round(accuracy_score(ref_labels, predicted_labels), 4)
        p, r, f, _         = precision_recall_fscore_support(
            ref_labels, predicted_labels, average="weighted", zero_division=0
        )
        result["precision"] = round(p, 4)
        result["recall"]    = round(r, 4)
        result["f1"]        = round(f, 4)
    except Exception as e:
        print(f"[!] Other metrics: {e}")
    return result


# ============================================================
# TRAINING
# ============================================================
def train_model(tokenized_train, tokenized_test, num_labels, output_dir, tokenizer):
    model = TwoTowerRoberta(
        model_name="FacebookAI/roberta-large",
        num_labels=num_labels,
        dropout=0.1,
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="cohen_kappa",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,   # required for custom model + collator
    )

    trainer = OrdinalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        data_collator=TwoTowerDataCollator(),
        sigma=1.0,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    best_model_path = trainer.state.best_model_checkpoint
    print(f"Best model checkpoint: {best_model_path}")

    # Save state dict + tokenizer for later reload
    if best_model_path:
        os.makedirs(best_model_path, exist_ok=True)
        torch.save(
            trainer.model.state_dict(),
            os.path.join(best_model_path, "pytorch_model.bin"),
        )
        tokenizer.save_pretrained(best_model_path)
        trainer.model.encoder.config.save_pretrained(best_model_path)

    return trainer, eval_results


# ============================================================
# DETAILED EVALUATION  — identical structure to original
# ============================================================
def detailed_evaluation(trainer, tokenized_valid, experiment_name: str = "experiment"):
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
        global_refs, global_predicted, target_names=LABEL_NAMES, zero_division=0
    ))

    global_acc        = accuracy_score(global_refs, global_predicted)
    global_qwk        = cohen_kappa_score(global_refs, global_predicted, weights="quadratic")
    global_lwk        = cohen_kappa_score(global_refs, global_predicted, weights="linear")
    global_pearson, _ = pearsonr(global_refs, global_predicted)
    print(f"✅ Accuracy:        {global_acc:.4f}")
    print(f"✅ QWK (quadratic): {global_qwk:.4f}")
    print(f"✅ LWK (linear):    {global_lwk:.4f}")
    print(f"✅ Pearson:         {global_pearson:.4f}")

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
    cm        = confusion_matrix(global_refs, global_predicted)
    cm_df     = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df.to_string())
    cm_df.to_csv(f"{out_dir}/confusion_matrix.csv")

    cm_norm    = cm.astype('float') / cm.sum(axis=1, keepdims=True)
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
        cls_preds    = global_predicted[mask]
        cls_refs     = global_refs[mask]
        n_total      = len(cls_refs)
        cls_acc      = (cls_preds == cls_refs).sum() / n_total
        cls_adj      = (np.abs(cls_preds.astype(int) - cls_refs.astype(int)) <= 1).mean()
        cls_mae      = np.abs(cls_preds.astype(int)  - cls_refs.astype(int)).mean()
        wrong_mask   = cls_preds != cls_refs
        top_conf_str = (
            ", ".join([
                f"{LABEL_NAMES[p]}({c})"
                for p, c in Counter(cls_preds[wrong_mask]).most_common(3)
            ]) if wrong_mask.sum() > 0 else "None"
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
            "mae":          round(mae_, 4),
            "n_samples":    len(sub_ds),
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
            "n_samples":    len(sub_ds),
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
        "text_a":          tokenized_valid["text_a"],
        "text_b":          tokenized_valid["text_b"],
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
        "accuracy":     round(global_acc,     4),
        "qwk":          round(global_qwk,     4),
        "lwk":          round(global_lwk,     4),
        "pearson":      round(global_pearson,  4),
        "adjacent_acc": round(global_adj_acc,  4),
        "mae":          round(float(global_mae), 4),
        "n_samples":    len(global_refs),
        "n_errors":     int(errors.sum()),
    }


# ============================================================
# S3 UPLOAD  (unchanged)
# ============================================================
'''
def upload_model_to_s3(local_dir, bucket_name, s3_prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative   = os.path.relpath(local_path, local_dir)
            s3_path    = os.path.join(s3_prefix, relative).replace("\", "/")
            print(f"Uploading {local_path} → s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)
'''

# ============================================================
# MAIN
# ============================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    # ── Step 1 : build / reload fixed eval sets ONCE ──────────────────────────
    val_ds, test_ds, eval_orig_indices, max_length = build_fixed_eval_sets(
        CSV_PATH, tokenizer, force_rebuild=True
    )

    # ── Step 2 : tokenize fixed eval sets ONCE ────────────────────────────────
    print(f"\nTokenizing fixed eval sets (max_length={max_length}) …")
    tokenized_test  = tokenize_ds(test_ds,  tokenizer, max_length)
    tokenized_valid = tokenize_ds(val_ds,   tokenizer, max_length)

    AGREEMENT_THRESHOLD = 60   # use all data, penalise low-agreement via weights

    experiments = [
        {"agreement_threshold": AGREEMENT_THRESHOLD,
         "name": f"two_tower_gte{AGREEMENT_THRESHOLD}"},
    ]

    os.makedirs("results", exist_ok=True)
    comparison = []

    for exp in experiments:
        threshold    = exp["agreement_threshold"]
        exp_name     = exp["name"]
        output_dir   = f"model_saved/roberta-large-two-tower-{exp_name}"
        exp_data_dir = f"data_v2/{exp_name}"

        print(f"\n{'='*80}")
        print(f"EXPERIMENT : {exp_name}  (agreement >= {threshold}%)")
        print('='*80)

        train_ds = prepare_train_set(
            CSV_PATH, eval_orig_indices, threshold, exp_data_dir
        )

        print("Tokenizing train set …")
        tokenized_train = tokenize_ds(train_ds, tokenizer, max_length)

        trainer, eval_results = train_model(
            tokenized_train, tokenized_test, NUM_LABELS, output_dir, tokenizer
        )
        print(f"\n[{exp_name}] Test-set eval results (early-stopping signal):")
        print(eval_results)

        summary = detailed_evaluation(
            trainer, tokenized_valid, experiment_name=exp_name
        )
        comparison.append(summary)

    # ── Step 4 : side-by-side comparison ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON  (identical val set for all experiments)")
    print("=" * 80)
    df_cmp = pd.DataFrame(comparison).set_index("experiment")
    print(df_cmp.T.to_string())
    df_cmp.to_csv("results/comparison_summary.csv")
    print("\nSaved → results/comparison_summary.csv")


if __name__ == "__main__":
    main()