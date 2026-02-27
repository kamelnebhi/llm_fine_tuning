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
CSV_PATH       = "data_v2/merged_accuracy_coherence_range_final.csv"
FIXED_EVAL_DIR = "data_v2/fixed_eval"
NUM_LABELS     = 6
TASK_NAMES     = ["accuracy", "coherence", "range"]
LABEL_NAMES    = [f"Score {i}" for i in range(NUM_LABELS)]

median_map = {
    "A1": 2, "A2": 5, "B1": 8, "B2": 11, "C1": 14, "C2": 16
}

session   = boto3.Session()
s3_client = session.client('s3')


# ============================================================
# MULTITASK CROSS-ENCODER MODEL
# ============================================================
class MultiTaskCrossEncoderRoberta(nn.Module):
    """
    Single-encoder multitask cross-encoder.
    Input  : [CLS] text_a [SEP] text_b [EOS]
    Output : logits concatenated → [batch, 3 * num_labels]
               [:, 0:6]   = accuracy
               [:, 6:12]  = coherence
               [:, 12:18] = range
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size   # 1024
        self.num_labels = num_labels

        # Shared projection — one representation feeds all 3 heads
        self.shared_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
        )

        # Task-specific heads
        self.head_accuracy  = nn.Linear(512, num_labels)
        self.head_coherence = nn.Linear(512, num_labels)
        self.head_range     = nn.Linear(512, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        sample_weight=None,
        **kwargs,
    ):
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]   # [CLS]

        shared = self.shared_projection(cls)   # [batch, 512]

        logits = torch.cat([
            self.head_accuracy(shared),    # [batch, 6]
            self.head_coherence(shared),   # [batch, 6]
            self.head_range(shared),       # [batch, 6]
        ], dim=-1)                         # [batch, 18]

        return SequenceClassifierOutput(loss=None, logits=logits)

# ============================================================
# DATA COLLATOR
# ============================================================
class MultiTaskDataCollator:
    WEIGHT_FIELDS = ["sample_weight_accuracy",
                     "sample_weight_coherence",
                     "sample_weight_range"]
    LABEL_FIELDS  = ["label_accuracy", "label_coherence", "label_range"]

    def __call__(self, features):
        batch = {}

        # input ids & attention mask
        for field in ["input_ids", "attention_mask"]:
            batch[field] = torch.tensor(
                [f[field] for f in features], dtype=torch.long
            )

        # un poids par tâche (absent en eval → ignoré)
        for field in self.WEIGHT_FIELDS:
            if field in features[0]:
                batch[field] = torch.tensor(
                    [f[field] for f in features], dtype=torch.float32
                )

        # labels [batch, 3]
        batch["labels"] = torch.tensor(
            [[f[lf] for lf in self.LABEL_FIELDS] for f in features],
            dtype=torch.long,
        )
        return batch


# ============================================================
# MULTITASK ORDINAL TRAINER
# ============================================================
class MultiTaskOrdinalTrainer(Trainer):
    def __init__(self, *args, sigma=None, task_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # sigma : float partagé OU liste [sigma_acc, sigma_coh, sigma_range]
        if sigma is None:
            sigma = [1.0, 1.5, 1.2]          # défaut recommandé
        elif isinstance(sigma, float):
            sigma = [sigma, sigma, sigma]
        self.sigma        = sigma
        self.task_weights = task_weights or [1.0, 1.5, 1.5]

    def _ordinal_loss(self, logits, labels, sample_weights=None, sigma=1.0):
        device  = logits.device
        classes = torch.arange(logits.shape[-1], device=device).float()

        soft_labels = torch.exp(
            -((classes.unsqueeze(0) - labels.unsqueeze(1).float()) ** 2)
            / (2 * sigma ** 2)
        )
        soft_labels     = soft_labels / soft_labels.sum(dim=-1, keepdim=True)
        log_probs       = torch.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_labels * log_probs).sum(dim=-1)

        if sample_weights is not None:
            return (per_sample_loss * sample_weights).mean()
        return per_sample_loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # récupérer les 3 poids (None en eval)
        sw_a = inputs.pop("sample_weight_accuracy",  None)
        sw_c = inputs.pop("sample_weight_coherence", None)
        sw_r = inputs.pop("sample_weight_range",     None)

        outputs = model(**inputs)
        logits  = outputs.logits
        device  = logits.device

        sw_a = sw_a.to(device) if sw_a is not None else None
        sw_c = sw_c.to(device) if sw_c is not None else None
        sw_r = sw_r.to(device) if sw_r is not None else None

        loss_a = self._ordinal_loss(logits[:, :NUM_LABELS],
                                    labels[:, 0], sw_a, sigma=self.sigma[0])
        loss_c = self._ordinal_loss(logits[:, NUM_LABELS:2*NUM_LABELS],
                                    labels[:, 1], sw_c, sigma=self.sigma[1])
        loss_r = self._ordinal_loss(logits[:, 2*NUM_LABELS:],
                                    labels[:, 2], sw_r, sigma=self.sigma[2])

        wa, wc, wr = self.task_weights
        loss = wa * loss_a + wc * loss_c + wr * loss_r

        return (loss, outputs) if return_outputs else loss


# ============================================================
# DATA PREPROCESSING — synthetic coherence & range labels
# ============================================================
def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['ef_level'] = df.apply(
        lambda row: median_map[row['cefr_level']] if pd.isna(row['ef_level'])
        else row['ef_level'],
        axis=1,
    )
    df['text_a'] = (
        "Prompt Level: " + df['ef_level'].astype(str)
        + " Prompt: "    + df['activity_instructions'].astype(str)
        + " Response: "  + df['student_submission'].astype(str)
    )
    df['text_b'] = df['correction'].astype(str)

    keep = ["orig_idx", "text_a", "text_b", "task_id", "ef_level",
            "majority_value_accuracy", "agreement_percentage_accuracy", 
            "majority_value_coherence", "agreement_percentage_coherence",
            "majority_value_range", "agreement_percentage_range"
            ]
    df = df[keep].rename(columns={
        'majority_value_accuracy': 'label_accuracy',
        'majority_value_coherence': 'label_coherence',
        'majority_value_range': 'label_range'
        })
    df.dropna(subset=['label_accuracy', 'label_coherence', 'label_range', 'text_a', 'text_b'], inplace=True)
    for col in ["label_accuracy", "label_coherence", "label_range"]:
        df[col] = df[col].round().astype("int64")
    df.reset_index(drop=True, inplace=True)

    # ── Synthetic labels : label_accuracy ± random {-1, 0, +1} ──────────
    # Replace this block once real coherence/range annotations are available
    '''
    rng     = np.random.default_rng(seed=42)
    noise_c = rng.choice([-1, 0, 1], size=len(df))
    noise_r = rng.choice([-1, 0, 1], size=len(df))
    df['label_coherence'] = (
        df['label_accuracy'] + noise_c
    ).clip(0, NUM_LABELS - 1).astype(int)
    df['label_range'] = (
        df['label_accuracy'] + noise_r
    ).clip(0, NUM_LABELS - 1).astype(int)
    '''
    return df


def _cast_labels(ds: Dataset) -> Dataset:
    feats = ds.features.copy()
    cl    = ClassLabel(names=[str(i) for i in range(NUM_LABELS)])
    feats["label_accuracy"]  = cl
    feats["label_coherence"] = cl
    feats["label_range"]     = cl
    return ds.cast(feats)


def load_jsonl_as_dataset(path: str) -> Dataset:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_pandas(pd.DataFrame(records))


# ============================================================
# TOKENIZATION — single cross-encoder input (unchanged)
# ============================================================
def tokenize_ds(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(examples):
        enc = tokenizer(
            examples["text_a"],
            examples["text_b"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
    return ds.map(_tokenize, batched=True)


# ============================================================
# FIXED EVAL SETS
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
    df["agreement_percentage_accuracy"] = pd.to_numeric(
        df["agreement_percentage_accuracy"], errors='coerce'
    )

    print("  Computing max_length from agreement>=60% pool …")
    df_pool      = df[df["agreement_percentage_accuracy"] >= 60]
    pair_lengths = tokenizer(
        df_pool['text_a'].tolist(),
        df_pool['text_b'].tolist(),
        truncation=False, padding=False, return_length=True,
    )["length"]
    p95        = int(np.percentile(pair_lengths, 95))
    max_length = min(max(p95, 128), 512)
    print(f"  Pair lengths — median={int(np.median(pair_lengths))}, 95th={p95}")
    print(f"  chosen max_length={max_length}")

    df_100   = df[df["agreement_percentage_accuracy"] == 100].copy()
    df_60_80 = df[
        (df["agreement_percentage_accuracy"] >= 60) &
        (df["agreement_percentage_accuracy"] <= 80)
    ].copy()

    EVAL_RATIO = 0.15

    def stratified_sample(df_in, ratio, seed):
        return (
            df_in.groupby('label_accuracy', group_keys=False)
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

    print(f"\n  Eval pool : {len(df_eval)} samples  "
          f"({n_from_100} agreement=100%, {n_from_noisy} agreement=60-80%)")

    ds_eval = _cast_labels(
        Dataset.from_pandas(
            df_eval[["orig_idx", "text_a", "text_b", "task_id", "ef_level",
                      "label_accuracy", "label_coherence", "label_range",
                      "is_100_agreement"]],
            preserve_index=False,
        )
    )
    split   = ds_eval.train_test_split(
        test_size=0.50, seed=42, stratify_by_column="label_accuracy"
    )
    val_ds  = split["train"]
    test_ds = split["test"]

    eval_orig_indices = set(val_ds["orig_idx"]) | set(test_ds["orig_idx"])

    print(f"  val={len(val_ds)}, test={len(test_ds)}")

    save_split_to_jsonl(val_ds,  val_path)
    save_split_to_jsonl(test_ds, test_path)

    meta = {
        "eval_orig_indices": list(eval_orig_indices),
        "max_length":        max_length,
        "n_val":             len(val_ds),
        "n_test":            len(test_ds),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → {FIXED_EVAL_DIR}/")
    return val_ds, test_ds, eval_orig_indices, max_length


# ============================================================
# TRAINING SET BUILDER
# ============================================================
def prepare_train_set(
    csv_path: str,
    eval_orig_indices: set,
    agreement_threshold: int,
    experiment_dir: str,
    threshold_accuracy: int  = None,
    threshold_coherence: int = None,
    threshold_range: int     = None,
) -> Dataset:
    
    def _compute_task_weight(df, agreement_col, label_col):
        agr = pd.to_numeric(df[agreement_col], errors='coerce').fillna(0.5) / 100.0
        label_counts = df[label_col].value_counts()
        total  = len(df)
        cw_map = {cls: total / (NUM_LABELS * cnt) for cls, cnt in label_counts.items()}
        sw = agr * df[label_col].map(cw_map)
        return (sw / sw.mean()).values

    # Si seuils spécifiques non fournis, fallback sur agreement_threshold global
    thr_acc = threshold_accuracy  if threshold_accuracy  is not None else agreement_threshold
    thr_coh = threshold_coherence if threshold_coherence is not None else agreement_threshold
    thr_rng = threshold_range     if threshold_range     is not None else agreement_threshold

    df_raw             = pd.read_csv(csv_path)
    df_raw['orig_idx'] = df_raw.index
    df                 = preprocess_df(df_raw)

    df["agreement_percentage_accuracy"]  = pd.to_numeric(df["agreement_percentage_accuracy"],  errors='coerce')
    df["agreement_percentage_coherence"] = pd.to_numeric(df["agreement_percentage_coherence"], errors='coerce')
    df["agreement_percentage_range"]     = pd.to_numeric(df["agreement_percentage_range"],     errors='coerce')

    df_thresh = df[
        (df["agreement_percentage_accuracy"]  >= thr_acc) &
        (df["agreement_percentage_coherence"] >= thr_coh) &
        (df["agreement_percentage_range"]     >= thr_rng)
    ]
    df_train  = df_thresh[~df_thresh['orig_idx'].isin(eval_orig_indices)].copy()
    df_train.reset_index(drop=True, inplace=True)

    if len(df_train) == 0:
        raise ValueError(
            f"Train set is empty for thresholds: "
            f"accuracy>={thr_acc}%, coherence>={thr_coh}%, range>={thr_rng}%."
        )

    # ── Sample weights indépendants par tâche ─────────────────────────────
    df_train["sample_weight_accuracy"]  = _compute_task_weight(
        df_train, "agreement_percentage_accuracy",  "label_accuracy"
    )
    df_train["sample_weight_coherence"] = _compute_task_weight(
        df_train, "agreement_percentage_coherence", "label_coherence"
    )
    df_train["sample_weight_range"]     = _compute_task_weight(
        df_train, "agreement_percentage_range",     "label_range"
    )

    print(f"\n[Train | accuracy>={thr_acc}% | coherence>={thr_coh}% | range>={thr_rng}%]"
          f"  n_samples={len(df_train)}")
    print("Label accuracy distribution:")
    print(df_train['label_accuracy'].value_counts().sort_index())
    print("Label coherence distribution:")
    print(df_train['label_coherence'].value_counts().sort_index())
    print("Label range distribution:")
    print(df_train['label_range'].value_counts().sort_index())

    print("\nSample weights stats:")
    for task in TASK_NAMES:
        col = f"sample_weight_{task}"
        print(f"  {task:>10} — mean={df_train[col].mean():.3f}  "
              f"min={df_train[col].min():.3f}  "
              f"max={df_train[col].max():.3f}")

    ds = _cast_labels(
        Dataset.from_pandas(
            df_train[["orig_idx", "text_a", "text_b", "task_id", "ef_level",
                       "label_accuracy", "label_coherence", "label_range",
                       "sample_weight_accuracy",
                       "sample_weight_coherence",
                       "sample_weight_range"]],
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
# METRICS — reported per task
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred   # logits: [n, 18],  labels: [n, 3]
    metrics = {}

    for i, task in enumerate(TASK_NAMES):
        task_logits = logits[:, i*NUM_LABELS:(i+1)*NUM_LABELS]
        task_preds  = np.argmax(task_logits, axis=-1)
        task_labels = labels[:, i]

        try:
            if len(set(task_labels)) > 1 and len(set(task_preds)) > 1:
                metrics[f"{task}_accuracy"]    = accuracy_score(task_labels, task_preds)
                metrics[f"{task}_cohen_kappa"] = cohen_kappa_score(
                    task_labels, task_preds, weights="quadratic"
                )
                metrics[f"{task}_pearson"], _  = pearsonr(task_labels, task_preds)
            else:
                metrics[f"{task}_accuracy"]    = float('nan')
                metrics[f"{task}_cohen_kappa"] = float('nan')
                metrics[f"{task}_pearson"]     = float('nan')
        except Exception as e:
            print(f"[!] Metrics error for {task}: {e}")

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
    model = MultiTaskCrossEncoderRoberta(
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
        metric_for_best_model="accuracy_cohen_kappa",   # monitor accuracy task QWK
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = MultiTaskOrdinalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        data_collator=MultiTaskDataCollator(),
        sigma=[0.8, 1.5, 1.2],          # acc=strict, coh=souple, range=moyen
        task_weights=[1.0, 1.5, 1.5],   # booster coh et range
    )

    trainer.train()
    eval_results = trainer.evaluate()

    best_model_path = trainer.state.best_model_checkpoint
    print(f"Best model checkpoint: {best_model_path}")

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
# DETAILED EVALUATION — per task
# ============================================================
def detailed_evaluation(trainer, tokenized_valid, experiment_name: str = "experiment"):
    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[{experiment_name}]  GLOBAL EVALUATION ON VALIDATION SET")
    print("=" * 80)

    global_preds = trainer.predict(tokenized_valid)
    all_logits   = global_preds.predictions   # [n, 18]
    all_labels   = global_preds.label_ids     # [n, 3]

    summary = {"experiment": experiment_name}

    for i, task in enumerate(TASK_NAMES):
        task_logits    = all_logits[:, i*NUM_LABELS:(i+1)*NUM_LABELS]
        task_predicted = np.argmax(task_logits, axis=-1)
        task_refs      = all_labels[:, i]

        print(f"\n{'─'*60}")
        print(f"  TASK : {task.upper()}")
        print(f"{'─'*60}")
        print(classification_report(
            task_refs, task_predicted, target_names=LABEL_NAMES, zero_division=0
        ))

        acc        = accuracy_score(task_refs, task_predicted)
        qwk        = cohen_kappa_score(task_refs, task_predicted, weights="quadratic")
        lwk        = cohen_kappa_score(task_refs, task_predicted, weights="linear")
        pearson, _ = pearsonr(task_refs, task_predicted)
        adj_acc    = (np.abs(task_predicted.astype(int) - task_refs.astype(int)) <= 1).mean()
        mae        = np.abs(task_predicted.astype(int) - task_refs.astype(int)).mean()

        print(f"  Accuracy     : {acc:.4f}")
        print(f"  QWK          : {qwk:.4f}")
        print(f"  LWK          : {lwk:.4f}")
        print(f"  Pearson      : {pearson:.4f}")
        print(f"  Adjacent Acc : {adj_acc:.4f}")
        print(f"  MAE          : {mae:.4f}")

        summary[f"{task}_accuracy"]     = round(acc,        4)
        summary[f"{task}_qwk"]          = round(qwk,        4)
        summary[f"{task}_lwk"]          = round(lwk,        4)
        summary[f"{task}_pearson"]      = round(pearson,    4)
        summary[f"{task}_adjacent_acc"] = round(adj_acc,    4)
        summary[f"{task}_mae"]          = round(float(mae), 4)

        # Confusion matrix per task
        from sklearn.metrics import confusion_matrix
        cm    = confusion_matrix(task_refs, task_predicted)
        cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
        cm_df.to_csv(f"{out_dir}/confusion_matrix_{task}.csv")

        # Normalised confusion matrix per task
        cm_norm    = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm_df = pd.DataFrame(
            np.round(cm_norm, 3), index=LABEL_NAMES, columns=LABEL_NAMES
        )
        cm_norm_df.to_csv(f"{out_dir}/confusion_matrix_{task}_normalized.csv")

    # ── Save full predictions (all 3 tasks per row) ───────────────────────
    rows = []
    for idx in range(len(tokenized_valid)):
        row = {
            "text_a":   tokenized_valid["text_a"][idx],
            "text_b":   tokenized_valid["text_b"][idx],
            "task_id":  tokenized_valid["task_id"][idx],
            "ef_level": tokenized_valid["ef_level"][idx],
        }
        for i, task in enumerate(TASK_NAMES):
            task_logits = all_logits[idx, i*NUM_LABELS:(i+1)*NUM_LABELS]
            probs       = torch.softmax(torch.tensor(task_logits), dim=-1).numpy()
            row[f"true_{task}"]       = int(all_labels[idx, i])
            row[f"predicted_{task}"]  = int(np.argmax(task_logits))
            row[f"confidence_{task}"] = round(float(probs.max()), 4)
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        f"{out_dir}/validation_predictions_full.csv", index=False
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SUMMARY — {experiment_name}")
    print("=" * 80)
    for task in TASK_NAMES:
        print(f"  [{task:>10}]  "
              f"acc={summary[f'{task}_accuracy']:.4f}  "
              f"qwk={summary[f'{task}_qwk']:.4f}  "
              f"pearson={summary[f'{task}_pearson']:.4f}  "
              f"adj={summary[f'{task}_adjacent_acc']:.4f}  "
              f"mae={summary[f'{task}_mae']:.4f}")
    print("=" * 80)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    val_ds, test_ds, eval_orig_indices, max_length = build_fixed_eval_sets(
        CSV_PATH, tokenizer, force_rebuild=True
    )

    print(f"\nTokenizing fixed eval sets (max_length={max_length}) …")
    tokenized_test  = tokenize_ds(test_ds,  tokenizer, max_length)
    tokenized_valid = tokenize_ds(val_ds,   tokenizer, max_length)

    AGREEMENT_THRESHOLD = 60

    experiments = [
        {"agreement_threshold": AGREEMENT_THRESHOLD,
         "name": f"multitask_cross_encoder_gte{AGREEMENT_THRESHOLD}"},
    ]

    os.makedirs("results", exist_ok=True)
    comparison = []

    for exp in experiments:
        threshold    = exp["agreement_threshold"]
        exp_name     = exp["name"]
        output_dir   = f"model_saved/roberta-large-multitask-{exp_name}"
        exp_data_dir = f"data_v2/{exp_name}"

        print(f"\n{'='*80}")
        print(f"EXPERIMENT : {exp_name}  (agreement >= {threshold}%)")
        print('='*80)

        train_ds = prepare_train_set(
            CSV_PATH,
            eval_orig_indices,
            agreement_threshold  = AGREEMENT_THRESHOLD,  # fallback global
            experiment_dir       = exp_data_dir,
            threshold_accuracy   = 60,
            threshold_coherence  = 50,
            threshold_range      = 50,
        )

        print("Tokenizing train set …")
        tokenized_train = tokenize_ds(train_ds, tokenizer, max_length)

        trainer, eval_results = train_model(
            tokenized_train, tokenized_test, NUM_LABELS, output_dir, tokenizer
        )
        print(f"\n[{exp_name}] Test-set eval results:")
        print(eval_results)

        summary = detailed_evaluation(
            trainer, tokenized_valid, experiment_name=exp_name
        )
        comparison.append(summary)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    df_cmp = pd.DataFrame(comparison).set_index("experiment")
    print(df_cmp.T.to_string())
    df_cmp.to_csv("results/comparison_summary.csv")
    print("\nSaved → results/comparison_summary.csv")


if __name__ == "__main__":
    main()
