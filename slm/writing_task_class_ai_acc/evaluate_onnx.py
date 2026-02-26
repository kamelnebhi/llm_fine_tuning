"""
evaluate_onnx.py
----------------
Evalue les modèles ONNX (original + quantifié) sur les mêmes
données de validation/test utilisées pendant l'entraînement.

Usage:
    python evaluate_onnx.py \
        --onnx_path      onnx_model/multitask_roberta.onnx \
        --quantized_path onnx_model/multitask_roberta_quantized.onnx \
        --eval_dir       data_v2/fixed_eval \
        --max_length     512 \
        --batch_size     16 \
        --output_dir     results/onnx_eval
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import onnx
if not hasattr(onnx, 'mapping'):
    import onnx._mapping as _m
    onnx.mapping = _m

import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import pearsonr
from datasets import Dataset
import torch

# ── constants (doivent correspondre à l'entraînement) ─────────────────────────
NUM_LABELS = 6
TASK_NAMES = ["accuracy", "coherence", "range"]
LABEL_NAMES = [f"Score {i}" for i in range(NUM_LABELS)]


# ============================================================
# UTILS
# ============================================================
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_jsonl(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


# ============================================================
# ORT SESSION
# ============================================================
def build_session(onnx_path: str) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


# ============================================================
# BATCH INFERENCE
# ============================================================
def run_inference(
    session: ort.InferenceSession,
    tokenizer,
    records: list,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """
    Returns logits array of shape [n_samples, 18].
    """
    all_logits = []

    for start in tqdm(range(0, len(records), batch_size), desc="Inference"):
        batch = records[start : start + batch_size]

        text_a = [r["text_a"] for r in batch]
        text_b = [r["text_b"] for r in batch]

        enc = tokenizer(
            text_a,
            text_b,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

        logits = session.run(["logits"], ort_inputs)[0]  # [batch, 18]
        all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)  # [n, 18]


# ============================================================
# METRICS
# ============================================================
def compute_task_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    metrics = {}

    if len(set(true_labels)) <= 1 or len(set(pred_labels)) <= 1:
        return {k: float("nan") for k in
                ["accuracy", "qwk", "lwk", "pearson",
                 "adjacent_acc", "mae", "precision", "recall", "f1"]}

    metrics["accuracy"]     = round(accuracy_score(true_labels, pred_labels), 4)
    metrics["qwk"]          = round(cohen_kappa_score(true_labels, pred_labels, weights="quadratic"), 4)
    metrics["lwk"]          = round(cohen_kappa_score(true_labels, pred_labels, weights="linear"), 4)
    metrics["pearson"]      = round(pearsonr(true_labels, pred_labels)[0], 4)
    metrics["adjacent_acc"] = round(
        (np.abs(pred_labels.astype(int) - true_labels.astype(int)) <= 1).mean(), 4
    )
    metrics["mae"]          = round(
        np.abs(pred_labels.astype(int) - true_labels.astype(int)).mean(), 4
    )

    p, r, f, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    metrics["precision"] = round(p, 4)
    metrics["recall"]    = round(r, 4)
    metrics["f1"]        = round(f, 4)

    return metrics


# ============================================================
# FULL EVALUATION — un modèle, un split
# ============================================================
def evaluate_model(
    session: ort.InferenceSession,
    tokenizer,
    records: list,
    max_length: int,
    batch_size: int,
    model_label: str,
    split_name: str,
    output_dir: str,
) -> dict:

    print(f"\n{'='*70}")
    print(f"  Model : {model_label}   |   Split : {split_name.upper()}")
    print(f"{'='*70}")

    # ── Inférence ──────────────────────────────────────────────────────────
    logits = run_inference(session, tokenizer, records, max_length, batch_size)
    # logits : [n, 18]

    # ── Récupérer les labels vrais ─────────────────────────────────────────
    true_accuracy  = np.array([r["label_accuracy"]  for r in records])
    true_coherence = np.array([r["label_coherence"] for r in records])
    true_range     = np.array([r["label_range"]     for r in records])
    true_labels    = np.stack([true_accuracy, true_coherence, true_range], axis=1)
    # true_labels : [n, 3]

    summary = {"model": model_label, "split": split_name}
    rows    = []

    for i, task in enumerate(TASK_NAMES):
        task_logits = logits[:, i * NUM_LABELS : (i + 1) * NUM_LABELS]
        probs       = softmax(task_logits)                  # [n, 6]
        preds       = np.argmax(probs, axis=-1)             # [n]
        trues       = true_labels[:, i]                     # [n]

        # ── métriques ──────────────────────────────────────────────────────
        m = compute_task_metrics(trues, preds)

        print(f"\n  {'─'*60}")
        print(f"  TASK : {task.upper()}")
        print(f"  {'─'*60}")
        print(classification_report(
            trues, preds, target_names=LABEL_NAMES, zero_division=0
        ))
        print(f"  Accuracy     : {m['accuracy']}")
        print(f"  QWK          : {m['qwk']}")
        print(f"  LWK          : {m['lwk']}")
        print(f"  Pearson      : {m['pearson']}")
        print(f"  Adjacent Acc : {m['adjacent_acc']}")
        print(f"  MAE          : {m['mae']}")

        for k, v in m.items():
            summary[f"{task}_{k}"] = v

        # ── confusion matrix ───────────────────────────────────────────────
        cm      = confusion_matrix(trues, preds, labels=list(range(NUM_LABELS)))
        cm_df   = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        tag = f"{model_label.replace(' ', '_')}_{split_name}_{task}"
        cm_df.to_csv(os.path.join(output_dir, f"cm_{tag}.csv"))
        pd.DataFrame(
            np.round(cm_norm, 3), index=LABEL_NAMES, columns=LABEL_NAMES
        ).to_csv(os.path.join(output_dir, f"cm_norm_{tag}.csv"))

        # ── per-sample rows ────────────────────────────────────────────────
        for idx in range(len(records)):
            rows.append({
                "idx":           idx,
                "task_id":       records[idx].get("task_id", ""),
                "ef_level":      records[idx].get("ef_level", ""),
                "task":          task,
                "true":          int(trues[idx]),
                "predicted":     int(preds[idx]),
                "confidence":    round(float(probs[idx].max()), 4),
                "abs_error":     int(abs(int(preds[idx]) - int(trues[idx]))),
                "model":         model_label,
                "split":         split_name,
            })

    # ── Sauvegarder les prédictions ────────────────────────────────────────
    pred_df = pd.DataFrame(rows)
    tag     = f"{model_label.replace(' ', '_')}_{split_name}"
    pred_df.to_csv(os.path.join(output_dir, f"predictions_{tag}.csv"), index=False)

    return summary


# ============================================================
# COMPARAISON ONNX vs QUANTIZED
# ============================================================
def compare_models(summaries: list, output_dir: str):
    df = pd.DataFrame(summaries)

    print("\n" + "=" * 70)
    print("COMPARAISON  ONNX FP32  vs  ONNX Int8 Quantifié")
    print("=" * 70)

    metrics_to_show = ["accuracy", "qwk", "pearson", "adjacent_acc", "mae"]

    for split in df["split"].unique():
        print(f"\n  Split : {split.upper()}")
        sub = df[df["split"] == split].set_index("model")

        for task in TASK_NAMES:
            print(f"\n    [{task}]")
            header = f"    {'Metric':<16}" + "".join(f"{m:<12}" for m in sub.index)
            print(header)
            for metric in metrics_to_show:
                col  = f"{task}_{metric}"
                vals = [str(sub.loc[m, col]) if col in sub.columns else "N/A"
                        for m in sub.index]
                print(f"    {metric:<16}" + "".join(f"{v:<12}" for v in vals))

    df.to_csv(os.path.join(output_dir, "onnx_eval_summary.csv"), index=False)
    print(f"\n  Résultats sauvegardés → {output_dir}/onnx_eval_summary.csv")


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path",      default="onnx_model/multitask_roberta.onnx")
    p.add_argument("--quantized_path", default="onnx_model/multitask_roberta_quantized.onnx")
    p.add_argument("--eval_dir",       default="data_v2/fixed_eval",
                   help="Dossier contenant validation.jsonl, test.jsonl et eval_meta.json")
    p.add_argument("--max_length",     type=int, default=None,
                   help="Si None, lu depuis eval_meta.json")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--output_dir",     default="results/onnx_eval")
    p.add_argument("--tokenizer",      default="FacebookAI/roberta-large")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Lire max_length depuis eval_meta.json si non fourni ───────────────
    meta_path = os.path.join(args.eval_dir, "eval_meta.json")
    if args.max_length is None:
        with open(meta_path) as f:
            meta = json.load(f)
        max_length = meta["max_length"]
        print(f"max_length lu depuis eval_meta.json : {max_length}")
    else:
        max_length = args.max_length

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ── Charger les splits ────────────────────────────────────────────────
    val_records  = load_jsonl(os.path.join(args.eval_dir, "validation.jsonl"))
    test_records = load_jsonl(os.path.join(args.eval_dir, "test.jsonl"))
    print(f"Validation : {len(val_records)} samples")
    print(f"Test       : {len(test_records)} samples")

    # ── Modèles à évaluer ─────────────────────────────────────────────────
    models_to_eval = []

    if os.path.exists(args.onnx_path):
        models_to_eval.append(("ONNX FP32",   args.onnx_path))
    else:
        print(f"[!] ONNX FP32 introuvable : {args.onnx_path}")

    if os.path.exists(args.quantized_path):
        models_to_eval.append(("ONNX Int8",   args.quantized_path))
    else:
        print(f"[!] ONNX quantifié introuvable : {args.quantized_path}")

    if not models_to_eval:
        raise FileNotFoundError("Aucun modèle ONNX trouvé.")

    # ── Évaluation ────────────────────────────────────────────────────────
    all_summaries = []

    for model_label, model_path in models_to_eval:
        print(f"\nChargement session : {model_label}  ({model_path})")
        session = build_session(model_path)

        for split_name, records in [("validation", val_records),
                                     ("test",       test_records)]:
            summary = evaluate_model(
                session     = session,
                tokenizer   = tokenizer,
                records     = records,
                max_length  = max_length,
                batch_size  = args.batch_size,
                model_label = model_label,
                split_name  = split_name,
                output_dir  = args.output_dir,
            )
            all_summaries.append(summary)

    # ── Comparaison finale ────────────────────────────────────────────────
    compare_models(all_summaries, args.output_dir)

    # ── Sauvegarder aussi en JSON ─────────────────────────────────────────
    with open(os.path.join(args.output_dir, "onnx_eval_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n✅ Évaluation terminée.")


if __name__ == "__main__":
    main()
