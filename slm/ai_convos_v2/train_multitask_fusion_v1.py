import os
os.environ["WANDB_DISABLED"] = "true"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
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
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────────
MERGED_CSV_PATH     = "data/merged_accuracy_coherence_range_ai_convos_v2.csv"
FEATURIZED_CSV_PATH = "data/featurized_sessions.csv"
FIXED_EVAL_DIR      = "data/fixed_eval_ai_convos"

# ── task config ────────────────────────────────────────────────────────────────
TASK_NAMES          = ["accuracy", "coherence", "range"]
# accuracy: 0-5 (6 classes), coherence: 0-4 (5), range: 0-4 (5)
NUM_LABELS_PER_TASK = {"accuracy": 6, "coherence": 5, "range": 5}
NUM_LABELS_TOTAL    = sum(NUM_LABELS_PER_TASK.values())   # 16

# Logit slicing — {task: (start, end)} over the 16-dim logit vector
LOGIT_OFFSETS: dict = {}
_off = 0
for _t in TASK_NAMES:
    LOGIT_OFFSETS[_t] = (_off, _off + NUM_LABELS_PER_TASK[_t])
    _off += NUM_LABELS_PER_TASK[_t]

MAX_LENGTH = 512

# ── tabular feature columns ────────────────────────────────────────────────────
AUDIO_COLS = [
    "mean_articulation_rate",  "mean_disfluency_rate",    "mean_fluency_rate",
    "mean_fragmentedness",     "mean_hesitation_ratio",   "mean_response_delay",
    "mean_silent_ratio",       "mean_speaking_confidence","mean_speaking_rate",
    "mean_speaking_rate_jitter","mean_speech_ratio",      "mean_spoken_duration",
]
_NLP_ID_COLS  = {"recordId", "roleplayid", "studentid"}
# Which aggregation suffixes to keep from featurized_sessions.
# Options: "mean", "std" — columns without a suffix (e.g. student_turn_count) are always kept.
NLP_AGG_MODES: list = ["mean"]   # default: mean only; set to ["mean", "std"] for both
NLP_COLS:      list = []         # filled by load_and_join_data()
TABULAR_COLS:  list = []         # AUDIO_COLS + NLP_COLS, filled by load_and_join_data()


# ============================================================
# ATTENTION POOLING
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores  = self.attn(hidden_states).squeeze(-1)
        scores  = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


# ============================================================
# TASK ADAPTER  (bottleneck per task on shared fused repr)
# ============================================================
class TaskAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up   = nn.Linear(bottleneck, hidden_size)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(x + self.up(self.act(self.down(x))))


# ============================================================
# TABULAR ENCODER  (audio + NLP session-level features)
# ============================================================
class TabularEncoder(nn.Module):
    """Two-layer MLP that projects tabular features to a fixed-dim repr."""
    def __init__(self, n_feats: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(n_feats, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MULTITASK FUSION MODEL  (RoBERTa + tabular → 3 MLP heads)
# ============================================================
class MultiTaskFusionRoberta(nn.Module):
    """
    Single-pass RoBERTa encoder fused with session-level audio+NLP features.
    Three asymmetric MLP heads for accuracy (0-5), coherence (0-4), range (0-4).
    """
    def __init__(
        self,
        model_name: str,
        num_labels_per_task: dict,
        n_tabular_feats: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder             = RobertaModel.from_pretrained(model_name)
        hidden_size              = self.encoder.config.hidden_size   # 1024 (large)
        trunk_dim                = 768
        tab_dim                  = 128
        fused_dim                = trunk_dim + tab_dim               # 896
        self.num_labels_per_task = num_labels_per_task

        # ── Text branch ───────────────────────────────────────────────────────
        self.attn_pool    = AttentionPooling(hidden_size)
        self.shared_trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
        )

        # ── Tabular branch ────────────────────────────────────────────────────
        self.tabular_enc = TabularEncoder(n_tabular_feats, tab_dim, dropout)

        # ── Per-task adapters on fused repr ───────────────────────────────────
        self.adapter_accuracy  = TaskAdapter(fused_dim, bottleneck=64)
        self.adapter_coherence = TaskAdapter(fused_dim, bottleneck=96)
        self.adapter_range     = TaskAdapter(fused_dim, bottleneck=80)

        # ── Asymmetric projections ────────────────────────────────────────────
        proj_dim_acc = 256
        proj_dim_coh = 384
        proj_dim_rng = 320
        self.proj_accuracy  = self._make_proj(fused_dim, proj_dim_acc, dropout)
        self.proj_coherence = self._make_proj(fused_dim, proj_dim_coh, dropout)
        self.proj_range     = self._make_proj(fused_dim, proj_dim_rng, dropout)

        # ── Classification heads ──────────────────────────────────────────────
        self.head_accuracy  = nn.Linear(proj_dim_acc, num_labels_per_task["accuracy"])
        self.head_coherence = nn.Linear(proj_dim_coh, num_labels_per_task["coherence"])
        self.head_range     = nn.Linear(proj_dim_rng, num_labels_per_task["range"])

        # ── Uncertainty weighting (learned per task) ──────────────────────────
        self.log_sigma = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _make_proj(in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tabular_feats=None,
        labels=None,
        **kwargs,
    ):
        # Text encoding (single pass)
        out       = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr  = out.last_hidden_state[:, 0, :]
        attn_repr = self.attn_pool(out.last_hidden_state, attention_mask)
        text_repr = self.shared_trunk(torch.cat([cls_repr, attn_repr], dim=-1))

        # Tabular encoding
        tab_repr = self.tabular_enc(tabular_feats)

        # Fusion
        fused = torch.cat([text_repr, tab_repr], dim=-1)

        # Per-task heads
        logits_acc = self.head_accuracy(
            self.proj_accuracy(self.adapter_accuracy(fused))
        )
        logits_coh = self.head_coherence(
            self.proj_coherence(self.adapter_coherence(fused))
        )
        logits_rng = self.head_range(
            self.proj_range(self.adapter_range(fused))
        )

        logits = torch.cat([logits_acc, logits_coh, logits_rng], dim=-1)
        return SequenceClassifierOutput(loss=None, logits=logits)


# ============================================================
# DATA COLLATOR
# ============================================================
class MultiTaskDataCollator:
    WEIGHT_FIELDS = [
        "sample_weight_accuracy",
        "sample_weight_coherence",
        "sample_weight_range",
    ]
    LABEL_FIELDS = ["label_accuracy", "label_coherence", "label_range"]

    def __call__(self, features):
        batch = {}

        batch["input_ids"] = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        batch["tabular_feats"] = torch.tensor(
            [f["tabular_feats"] for f in features], dtype=torch.float32
        )

        for field in self.WEIGHT_FIELDS:
            if field in features[0]:
                batch[field] = torch.tensor(
                    [f[field] for f in features], dtype=torch.float32
                )

        batch["labels"] = torch.tensor(
            [[f[lf] for lf in self.LABEL_FIELDS] for f in features],
            dtype=torch.long,
        )
        return batch


# ============================================================
# MULTITASK ORDINAL TRAINER  (uncertainty-weighted losses)
# ============================================================
class MultiTaskOrdinalTrainer(Trainer):

    def __init__(self, *args, sigma=None, encoder_lr=1e-5, head_lr=3e-4, **kwargs):
        super().__init__(*args, **kwargs)
        if sigma is None:
            sigma = [0.7, 0.9, 0.85]
        elif isinstance(sigma, float):
            sigma = [sigma] * 3
        self.sigma      = sigma
        self.encoder_lr = encoder_lr
        self.head_lr    = head_lr

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        encoder_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.encoder_lr, "weight_decay": 0.01},
            {"params": head_params,    "lr": self.head_lr,    "weight_decay": 0.001},
        ])
        return self.optimizer

    def _ordinal_loss(self, logits, labels, sample_weights=None, sigma=0.8):
        device      = logits.device
        classes     = torch.arange(logits.shape[-1], device=device).float()
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
        sw_a   = inputs.pop("sample_weight_accuracy",  None)
        sw_c   = inputs.pop("sample_weight_coherence", None)
        sw_r   = inputs.pop("sample_weight_range",     None)

        outputs = model(**inputs)
        logits  = outputs.logits
        device  = logits.device

        sw_a = sw_a.to(device) if sw_a is not None else None
        sw_c = sw_c.to(device) if sw_c is not None else None
        sw_r = sw_r.to(device) if sw_r is not None else None

        s_a, e_a = LOGIT_OFFSETS["accuracy"]
        s_c, e_c = LOGIT_OFFSETS["coherence"]
        s_r, e_r = LOGIT_OFFSETS["range"]

        loss_a = self._ordinal_loss(logits[:, s_a:e_a], labels[:, 0], sw_a, self.sigma[0])
        loss_c = self._ordinal_loss(logits[:, s_c:e_c], labels[:, 1], sw_c, self.sigma[1])
        loss_r = self._ordinal_loss(logits[:, s_r:e_r], labels[:, 2], sw_r, self.sigma[2])

        log_sigma = model.log_sigma
        loss = (
              torch.exp(-2 * log_sigma[0]) * loss_a + log_sigma[0]
            + torch.exp(-2 * log_sigma[1]) * loss_c + log_sigma[1]
            + torch.exp(-2 * log_sigma[2]) * loss_r + log_sigma[2]
        )

        if self.state.global_step % 100 == 0 and model.training:
            s = torch.exp(log_sigma).detach().cpu().numpy()
            w = torch.exp(-2 * log_sigma).detach().cpu().numpy()
            print(
                f"\n  [step {self.state.global_step}] "
                f"sigmas acc={s[0]:.3f} coh={s[1]:.3f} rng={s[2]:.3f} | "
                f"weights acc={w[0]:.3f} coh={w[1]:.3f} rng={w[2]:.3f} | "
                f"losses acc={loss_a.item():.4f} coh={loss_c.item():.4f} rng={loss_r.item():.4f}"
            )

        return (loss, outputs) if return_outputs else loss


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_and_join_data() -> pd.DataFrame:
    """Load both CSVs, determine NLP/TABULAR cols, inner-join on recordId."""
    global NLP_COLS, TABULAR_COLS

    df_merged = pd.read_csv(MERGED_CSV_PATH)
    df_feat   = pd.read_csv(FEATURIZED_CSV_PATH)

    # Keep columns whose suffix matches NLP_AGG_MODES, plus non-aggregated cols
    # (e.g. student_turn_count has no _mean/_std suffix and is always included).
    NLP_COLS = [
        c for c in df_feat.columns
        if c not in _NLP_ID_COLS
        and (
            any(c.endswith(f"_{agg}") for agg in NLP_AGG_MODES)
            or not (c.endswith("_mean") or c.endswith("_std"))
        )
    ]
    TABULAR_COLS = AUDIO_COLS + NLP_COLS

    # Drop redundant ID cols from featurized before merging
    df_feat_clean = df_feat.drop(
        columns=[c for c in ["roleplayid", "studentid"] if c in df_feat.columns],
        errors="ignore",
    )

    df = df_merged.merge(df_feat_clean, on="recordId", how="inner")
    print(
        f"Data loaded: {len(df_merged)} merged x {len(df_feat)} featurized "
        f"-> {len(df)} joined (inner on recordId)"
    )
    print(f"Tabular: {len(AUDIO_COLS)} audio + {len(NLP_COLS)} NLP = {len(TABULAR_COLS)} total")
    return df


def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build text_input from roleplay context + conversation,
    rename and validate labels, clean tabular features.
    """
    df = df_raw.copy()

    # Build text input: short roleplay header + full conversation
    df["text_input"] = (
        "Roleplay: "       + df["rp_title"].fillna("").astype(str)
        + " | Difficulty: " + df["rp_difficulty"].fillna("").astype(str)
        + " | Your role: "  + df["rp_user_role"].fillna("").astype(str)
        + "\n"              + df["full_conversation"].fillna("").astype(str)
    )

    df = df.rename(columns={
        "majority_value_accuracy":  "label_accuracy",
        "majority_value_coherence": "label_coherence",
        "majority_value_range":     "label_range",
    })

    for col in ["label_accuracy", "label_coherence", "label_range"]:
        df[col] = df[col].round().astype("int64")

    # Clip to valid label ranges
    df = df[df["label_accuracy"].between(0, NUM_LABELS_PER_TASK["accuracy"] - 1)]
    df = df[df["label_coherence"].between(0, NUM_LABELS_PER_TASK["coherence"] - 1)]
    df = df[df["label_range"].between(0, NUM_LABELS_PER_TASK["range"] - 1)]

    # Clean tabular: replace inf, fill NaN with column median
    for col in TABULAR_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            med = df[col].median()
            df[col] = df[col].fillna(med if not np.isnan(med) else 0.0)

    df.dropna(
        subset=["label_accuracy", "label_coherence", "label_range", "text_input"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def _cast_labels(ds: Dataset) -> Dataset:
    feats = ds.features.copy()
    for task in TASK_NAMES:
        feats[f"label_{task}"] = ClassLabel(
            names=[str(i) for i in range(NUM_LABELS_PER_TASK[task])]
        )
    return ds.cast(feats)


def load_jsonl_as_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_pandas(pd.DataFrame(records))


def save_split_to_jsonl(dataset_split, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for record in dataset_split:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_ds(ds: Dataset, tokenizer, max_length: int = MAX_LENGTH) -> Dataset:
    def _tokenize(examples):
        enc = tokenizer(
            examples["text_input"],
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
# SEQUENCE LENGTH ANALYSIS
# ============================================================
def analyze_sequence_lengths(df_full: pd.DataFrame, tokenizer) -> int:
    """
    Tokenize all text_input sequences (no padding/truncation) and report
    length percentiles. Returns the recommended max_length (p95, capped at 512).
    """
    df = preprocess_df(df_full)
    texts = df["text_input"].tolist()

    print(f"\nAnalyzing token lengths on {len(texts)} sequences ...")
    lengths = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_length=True,
        add_special_tokens=True,
    )["length"]
    lengths = np.array(lengths)

    percentiles = [50, 75, 90, 95, 99, 100]
    print(f"\n  {'Percentile':>12}  {'Tokens':>8}")
    print(f"  {'-'*22}")
    for p in percentiles:
        print(f"  {f'p{p}':>12}  {int(np.percentile(lengths, p)):>8}")
    print(f"  {'mean':>12}  {lengths.mean():>8.1f}")

    p95         = int(np.percentile(lengths, 95))
    recommended = min(max(p95, 128), 512)
    truncated   = (lengths > recommended).sum()

    print(f"\n  Recommended max_length : {recommended}  (p95={p95})")
    print(f"  Sequences truncated    : {truncated} / {len(lengths)} "
          f"({100 * truncated / len(lengths):.1f}%)")

    if recommended < 512:
        print(f"  -> 512 is safe but {recommended} would cover 95% with less compute.")
    else:
        print("  -> 512 is necessary to cover 95% of sequences.")

    return recommended


# ============================================================
# SCALER
# ============================================================
def apply_scaler(ds: Dataset, scaler: StandardScaler) -> Dataset:
    """Normalise tabular_feats column using a pre-fitted StandardScaler."""
    def _scale(examples):
        arr    = np.array(examples["tabular_feats"], dtype=np.float32)
        scaled = scaler.transform(arr)
        return {"tabular_feats": scaled.tolist()}
    return ds.map(_scale, batched=True)


# ============================================================
# FIXED EVAL SETS
# ============================================================
def build_fixed_eval_sets(
    df_full: pd.DataFrame,
    force_rebuild: bool           = False,
    eval_threshold_accuracy:  int = 60,
    eval_threshold_coherence: int = 60,
    eval_threshold_range:     int = 60,
):
    """
    Build stable val/test sets filtered on annotation agreement.
    Tabular features are stored RAW (unscaled) — caller must apply_scaler()
    after fitting the scaler on train data.

    Returns: (val_ds, test_ds, eval_record_ids: set[str])
    """
    os.makedirs(FIXED_EVAL_DIR, exist_ok=True)
    meta_path = os.path.join(FIXED_EVAL_DIR, "eval_meta.json")
    val_path  = os.path.join(FIXED_EVAL_DIR, "validation.jsonl")
    test_path = os.path.join(FIXED_EVAL_DIR, "test.jsonl")

    if not force_rebuild and os.path.exists(meta_path):
        print("  Loading existing fixed eval sets from disk ...")
        with open(meta_path) as f:
            meta = json.load(f)
        val_ds  = _cast_labels(load_jsonl_as_dataset(val_path))
        test_ds = _cast_labels(load_jsonl_as_dataset(test_path))
        eval_record_ids = set(meta["eval_record_ids"])
        print(f"  val={len(val_ds)}, test={len(test_ds)}")
        return val_ds, test_ds, eval_record_ids

    print("Building fixed eval sets ...")
    df = preprocess_df(df_full)

    for col in ["agreement_percentage_accuracy",
                "agreement_percentage_coherence",
                "agreement_percentage_range"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reliability filter: all three tasks must meet the threshold
    df_reliable = df[
        (df["agreement_percentage_accuracy"]  >= eval_threshold_accuracy)  &
        (df["agreement_percentage_coherence"] >= eval_threshold_coherence) &
        (df["agreement_percentage_range"]     >= eval_threshold_range)
    ].copy()
    print(
        f"  Reliability filter (acc>={eval_threshold_accuracy}%, "
        f"coh>={eval_threshold_coherence}%, rng>={eval_threshold_range}%): "
        f"{len(df_reliable)} samples"
    )

    # Stratified pool: 60% unanimous + 40% contested
    df_100   = df_reliable[df_reliable["agreement_percentage_accuracy"] == 100].copy()
    df_noisy = df_reliable[
        (df_reliable["agreement_percentage_accuracy"] >= eval_threshold_accuracy) &
        (df_reliable["agreement_percentage_accuracy"] <  100)
    ].copy()

    EVAL_RATIO = 0.15

    def _strat_sample(df_in, ratio, seed):
        if len(df_in) == 0:
            return df_in
        return (
            df_in.groupby("label_accuracy", group_keys=False)
                 .apply(lambda x: x.sample(max(1, int(len(x) * ratio)), random_state=seed))
        )

    s100   = _strat_sample(df_100,   EVAL_RATIO, seed=42)
    snoisy = _strat_sample(df_noisy, EVAL_RATIO, seed=42)

    n_total      = len(s100) + len(snoisy)
    n_from_100   = min(int(n_total * 0.60), len(s100))
    n_from_noisy = min(n_total - n_from_100, len(snoisy))

    final_100   = s100.sample(n=n_from_100,   random_state=42) if n_from_100   > 0 else s100
    final_noisy = snoisy.sample(n=n_from_noisy, random_state=42) if n_from_noisy > 0 else snoisy

    df_eval = pd.concat([final_100, final_noisy], ignore_index=True)
    df_eval["is_100_agreement"] = df_eval["recordId"].isin(final_100["recordId"])

    print(
        f"  Eval pool: {len(df_eval)} "
        f"({n_from_100} unanimous, {n_from_noisy} contested)"
    )

    # Store raw (unscaled) tabular features
    df_eval["tabular_feats"] = df_eval[TABULAR_COLS].values.tolist()

    keep = [
        "recordId", "text_input", "rp_title", "rp_difficulty",
        "label_accuracy", "label_coherence", "label_range",
        "tabular_feats", "is_100_agreement",
    ]
    ds_eval = _cast_labels(
        Dataset.from_pandas(df_eval[keep], preserve_index=False)
    )

    split   = ds_eval.train_test_split(
        test_size=0.50, seed=42, stratify_by_column="label_accuracy"
    )
    val_ds  = split["train"]
    test_ds = split["test"]

    eval_record_ids = set(val_ds["recordId"]) | set(test_ds["recordId"])
    print(f"  val={len(val_ds)}, test={len(test_ds)}")

    save_split_to_jsonl(val_ds,  val_path)
    save_split_to_jsonl(test_ds, test_path)

    meta = {
        "eval_record_ids":          list(eval_record_ids),
        "n_val":                     len(val_ds),
        "n_test":                    len(test_ds),
        "eval_threshold_accuracy":   eval_threshold_accuracy,
        "eval_threshold_coherence":  eval_threshold_coherence,
        "eval_threshold_range":      eval_threshold_range,
        "n_tabular_feats":           len(TABULAR_COLS),
        "tabular_cols":              TABULAR_COLS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved -> {FIXED_EVAL_DIR}/")

    return val_ds, test_ds, eval_record_ids


# ============================================================
# TRAINING SET BUILDER
# ============================================================
def prepare_train_set(
    df_full: pd.DataFrame,
    eval_record_ids: set,
    agreement_threshold: int,
    experiment_dir: str,
    threshold_accuracy:  int = None,
    threshold_coherence: int = None,
    threshold_range:     int = None,
) -> tuple:
    """
    Build training dataset, fit StandardScaler on tabular features.
    Returns (train_ds, fitted_scaler).
    """
    def _compute_task_weight(df, agreement_col, label_col):
        agr = (
            pd.to_numeric(df[agreement_col], errors="coerce").fillna(0.5) / 100.0
        )
        label_counts = df[label_col].value_counts()
        n_cls  = len(label_counts)
        total  = len(df)
        cw_map = {cls: total / (n_cls * cnt) for cls, cnt in label_counts.items()}
        sw     = agr * df[label_col].map(cw_map)
        sw     = (sw / sw.mean()).values
        return np.clip(sw, 0.1, 10.0)

    thr_acc = threshold_accuracy  or agreement_threshold
    thr_coh = threshold_coherence or agreement_threshold
    thr_rng = threshold_range     or agreement_threshold

    df = preprocess_df(df_full)

    for col in ["agreement_percentage_accuracy",
                "agreement_percentage_coherence",
                "agreement_percentage_range"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_thresh = df[
        (df["agreement_percentage_accuracy"]  >= thr_acc) &
        (df["agreement_percentage_coherence"] >= thr_coh) &
        (df["agreement_percentage_range"]     >= thr_rng)
    ]
    df_train = df_thresh[~df_thresh["recordId"].isin(eval_record_ids)].copy()
    df_train.reset_index(drop=True, inplace=True)

    if len(df_train) == 0:
        raise ValueError(
            f"Train set empty for thresholds "
            f"acc>={thr_acc}%, coh>={thr_coh}%, rng>={thr_rng}%"
        )

    print(
        f"\n[Train | acc>={thr_acc}% | coh>={thr_coh}% | rng>={thr_rng}%] "
        f"n={len(df_train)}"
    )
    for task in TASK_NAMES:
        print(f"\nLabel {task} distribution:")
        print(df_train[f"label_{task}"].value_counts().sort_index())

    # Sample weights per task
    df_train["sample_weight_accuracy"]  = _compute_task_weight(
        df_train, "agreement_percentage_accuracy",  "label_accuracy"
    )
    df_train["sample_weight_coherence"] = _compute_task_weight(
        df_train, "agreement_percentage_coherence", "label_coherence"
    )
    df_train["sample_weight_range"]     = _compute_task_weight(
        df_train, "agreement_percentage_range",     "label_range"
    )

    # Fit scaler on training tabular features only
    tabular_arr = df_train[TABULAR_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(tabular_arr)
    df_train["tabular_feats"] = scaler.transform(tabular_arr).tolist()

    print("\nSample weight stats:")
    for task in TASK_NAMES:
        col = f"sample_weight_{task}"
        print(f"  {task:>10} — mean={df_train[col].mean():.3f}  "
              f"min={df_train[col].min():.3f}  max={df_train[col].max():.3f}")

    keep = [
        "recordId", "text_input", "rp_title", "rp_difficulty",
        "label_accuracy", "label_coherence", "label_range",
        "tabular_feats",
        "sample_weight_accuracy", "sample_weight_coherence", "sample_weight_range",
    ]
    ds = _cast_labels(
        Dataset.from_pandas(df_train[keep], preserve_index=False)
    )

    os.makedirs(experiment_dir, exist_ok=True)
    save_split_to_jsonl(ds, os.path.join(experiment_dir, "train.jsonl"))

    scaler_path = os.path.join(experiment_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved -> {scaler_path}")

    return ds, scaler


# ============================================================
# METRICS
# ============================================================
def compute_metrics(eval_pred):
    """
    Harmonic-mean composite QWK across 3 tasks.
    Logits: [n, 16] — sliced via LOGIT_OFFSETS.
    Labels: [n, 3]  — [accuracy, coherence, range].
    """
    logits, labels = eval_pred
    metrics    = {}
    qwk_scores = []

    for i, task in enumerate(TASK_NAMES):
        s, e = LOGIT_OFFSETS[task]
        task_logits = logits[:, s:e]
        task_preds  = np.argmax(task_logits, axis=-1)
        task_labels = labels[:, i]

        try:
            if len(set(task_labels)) > 1 and len(set(task_preds)) > 1:
                acc     = accuracy_score(task_labels, task_preds)
                qwk     = cohen_kappa_score(task_labels, task_preds, weights="quadratic")
                prs, _  = pearsonr(task_labels.astype(float), task_preds.astype(float))
                adj_acc = (np.abs(task_preds.astype(int) - task_labels.astype(int)) <= 1).mean()
                mae     = np.abs(task_preds.astype(int) - task_labels.astype(int)).mean()

                metrics[f"{task}_accuracy"]    = float(acc)
                metrics[f"{task}_cohen_kappa"] = float(qwk)
                metrics[f"{task}_pearson"]     = float(prs)
                metrics[f"{task}_adj_acc"]     = float(adj_acc)
                metrics[f"{task}_mae"]         = float(mae)
                qwk_scores.append(qwk)
            else:
                for m in ["_accuracy", "_cohen_kappa", "_pearson", "_adj_acc", "_mae"]:
                    metrics[f"{task}{m}"] = float("nan")
        except Exception as exc:
            print(f"[!] Metrics error for {task}: {exc}")

    # Harmonic mean of QWK — penalises weak tasks
    if qwk_scores and all(q > 0 for q in qwk_scores):
        metrics["composite_qwk"] = float(
            len(qwk_scores) / sum(1.0 / max(q, 0.01) for q in qwk_scores)
        )
    else:
        metrics["composite_qwk"] = float(np.mean(qwk_scores) if qwk_scores else float("nan"))

    return metrics


# ============================================================
# TRAINING
# ============================================================
def train_model(
    tokenized_train,
    tokenized_test,
    n_tabular_feats: int,
    output_dir: str,
    tokenizer,
):
    model = MultiTaskFusionRoberta(
        model_name="FacebookAI/roberta-large",
        num_labels_per_task=NUM_LABELS_PER_TASK,
        n_tabular_feats=n_tabular_feats,
        dropout=0.1,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        learning_rate=2e-5,         # fallback (overridden by custom optimizer)
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,   # effective batch = 64
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="composite_qwk",
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
        sigma=[0.7, 0.9, 0.85],
        encoder_lr=1e-5,
        head_lr=3e-4,
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
# DETAILED EVALUATION
# ============================================================
def detailed_evaluation(
    trainer,
    tokenized_valid,
    experiment_name: str = "experiment",
) -> dict:
    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[{experiment_name}]  EVALUATION ON VALIDATION SET")
    print("=" * 80)

    global_preds = trainer.predict(tokenized_valid)
    all_logits   = global_preds.predictions   # [n, 16]
    all_labels   = global_preds.label_ids     # [n, 3]

    summary = {"experiment": experiment_name}

    for i, task in enumerate(TASK_NAMES):
        s, e = LOGIT_OFFSETS[task]
        n_labels       = NUM_LABELS_PER_TASK[task]
        label_names    = [f"Score {j}" for j in range(n_labels)]
        task_logits    = all_logits[:, s:e]
        task_predicted = np.argmax(task_logits, axis=-1)
        task_refs      = all_labels[:, i]

        print(f"\n{'─' * 60}")
        print(f"  TASK : {task.upper()}  (labels 0-{n_labels - 1})")
        print(f"{'─' * 60}")
        print(classification_report(
            task_refs, task_predicted,
            labels=list(range(n_labels)),
            target_names=label_names,
            zero_division=0,
        ))

        acc        = accuracy_score(task_refs, task_predicted)
        qwk        = cohen_kappa_score(task_refs, task_predicted, weights="quadratic")
        lwk        = cohen_kappa_score(task_refs, task_predicted, weights="linear")
        pearson, _ = pearsonr(task_refs.astype(float), task_predicted.astype(float))
        adj_acc    = (np.abs(task_predicted.astype(int) - task_refs.astype(int)) <= 1).mean()
        mae        = np.abs(task_predicted.astype(int) - task_refs.astype(int)).mean()

        print(f"  Exact Accuracy : {acc:.4f}")
        print(f"  QWK            : {qwk:.4f}")
        print(f"  LWK            : {lwk:.4f}")
        print(f"  Pearson r      : {pearson:.4f}")
        print(f"  Adjacent Acc   : {adj_acc:.4f}  (|pred - true| <= 1)")
        print(f"  MAE            : {mae:.4f}")

        summary[f"{task}_accuracy"]     = round(acc,        4)
        summary[f"{task}_qwk"]          = round(qwk,        4)
        summary[f"{task}_lwk"]          = round(lwk,        4)
        summary[f"{task}_pearson"]      = round(pearson,    4)
        summary[f"{task}_adjacent_acc"] = round(adj_acc,    4)
        summary[f"{task}_mae"]          = round(float(mae), 4)

        cm     = confusion_matrix(task_refs, task_predicted, labels=list(range(n_labels)))
        cm_df  = pd.DataFrame(cm, index=label_names, columns=label_names)
        cm_df.to_csv(f"{out_dir}/confusion_matrix_{task}.csv")

        cm_norm    = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm_df = pd.DataFrame(np.round(cm_norm, 3), index=label_names, columns=label_names)
        cm_norm_df.to_csv(f"{out_dir}/confusion_matrix_{task}_normalized.csv")

    # ── Full predictions CSV ──────────────────────────────────────────────────
    rows = []
    for idx in range(len(tokenized_valid)):
        row = {
            "rp_title":     tokenized_valid["rp_title"][idx],
            "rp_difficulty": tokenized_valid["rp_difficulty"][idx],
        }
        for i, task in enumerate(TASK_NAMES):
            s, e = LOGIT_OFFSETS[task]
            tl    = all_logits[idx, s:e]
            probs = torch.softmax(torch.tensor(tl), dim=-1).numpy()
            row[f"true_{task}"]       = int(all_labels[idx, i])
            row[f"predicted_{task}"]  = int(np.argmax(tl))
            row[f"confidence_{task}"] = round(float(probs.max()), 4)
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        f"{out_dir}/validation_predictions_full.csv", index=False
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SUMMARY — {experiment_name}")
    print("=" * 80)
    for task in TASK_NAMES:
        print(
            f"  [{task:>10}]  "
            f"acc={summary[f'{task}_accuracy']:.4f}  "
            f"qwk={summary[f'{task}_qwk']:.4f}  "
            f"pearson={summary[f'{task}_pearson']:.4f}  "
            f"adj={summary[f'{task}_adjacent_acc']:.4f}  "
            f"mae={summary[f'{task}_mae']:.4f}"
        )
    print("=" * 80)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    # ── 1. Load & join both CSV files ────────────────────────────────────────
    df_full   = load_and_join_data()
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    # ── 2. Sequence length analysis (informs MAX_LENGTH choice) ──────────────
    analyze_sequence_lengths(df_full, tokenizer)

    # ── 3. Build stable eval sets (raw unscaled tabular) ─────────────────────
    val_ds, test_ds, eval_record_ids = build_fixed_eval_sets(
        df_full,
        force_rebuild=True,
        eval_threshold_accuracy=60,
        eval_threshold_coherence=60,
        eval_threshold_range=60,
    )

    experiments = [{"name": "multitask_fusion_roberta_v1"}]
    os.makedirs("results", exist_ok=True)
    comparison = []

    for exp in experiments:
        exp_name   = exp["name"]
        output_dir = f"model_saved/roberta-large-fusion-{exp_name}"

        # ── 4. Build train set + fit scaler on train tabular features ─────────
        train_ds, scaler = prepare_train_set(
            df_full,
            eval_record_ids,
            agreement_threshold=50,
            experiment_dir=f"data/{exp_name}",
            threshold_accuracy=50,
            threshold_coherence=50,
            threshold_range=50,
        )

        # ── 5. Apply fitted scaler to eval sets (no leakage) ─────────────────
        val_ds_scaled  = apply_scaler(val_ds,  scaler)
        test_ds_scaled = apply_scaler(test_ds, scaler)

        # ── 6. Tokenize all splits ────────────────────────────────────────────
        n_tabular = len(TABULAR_COLS)
        print(f"\nTokenizing (max_length={MAX_LENGTH}, tabular={n_tabular}) ...")
        tokenized_train = tokenize_ds(train_ds,       tokenizer)
        tokenized_test  = tokenize_ds(test_ds_scaled, tokenizer)
        tokenized_valid = tokenize_ds(val_ds_scaled,  tokenizer)

        # ── 7. Train ──────────────────────────────────────────────────────────
        trainer, eval_results = train_model(
            tokenized_train, tokenized_test,
            n_tabular_feats=n_tabular,
            output_dir=output_dir,
            tokenizer=tokenizer,
        )

        # ── 8. Evaluate on validation set ─────────────────────────────────────
        summary = detailed_evaluation(trainer, tokenized_valid, experiment_name=exp_name)
        comparison.append(summary)

    df_cmp = pd.DataFrame(comparison).set_index("experiment")
    df_cmp.to_csv("results/comparison_summary.csv")
    print("\nComparison saved -> results/comparison_summary.csv")


if __name__ == "__main__":
    main()
