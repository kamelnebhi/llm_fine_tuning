"""
convert_to_onnx.py
------------------
Converts a trained MultiTaskCrossEncoderRoberta checkpoint to ONNX,
then runs a quick inference test to validate the export.

Usage:
    python convert_to_onnx.py \
        --checkpoint model_saved/roberta-large-multitask-multitask_cross_encoder_gte60/checkpoint-600 \
        --output_dir onnx_model \
        --max_length 512
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from transformers import AutoTokenizer, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from onnxruntime.quantization import quantize_dynamic, QuantType

# ── constants (must match training script) ─────────────────────────────────────
NUM_LABELS = 6
TASK_NAMES = ["accuracy", "coherence", "range"]


# ============================================================
# MODEL DEFINITION  (copy from training script)
# ============================================================
class MultiTaskCrossEncoderRoberta(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.num_labels = num_labels

        self.shared_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
        )
        self.head_accuracy  = nn.Linear(512, num_labels)
        self.head_coherence = nn.Linear(512, num_labels)
        self.head_range     = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]

        shared = self.shared_projection(cls)

        logits = torch.cat([
            self.head_accuracy(shared),
            self.head_coherence(shared),
            self.head_range(shared),
        ], dim=-1)

        return logits   # [batch, 18]  — plain tensor, no wrapper


# ============================================================
# ONNX WRAPPER
# — Dropout must be disabled; we wrap forward() cleanly
# ============================================================
class OnnxExportWrapper(nn.Module):
    """Thin wrapper that accepts (input_ids, attention_mask) and
    returns a single tensor — required by torch.onnx.export."""

    def __init__(self, model: MultiTaskCrossEncoderRoberta):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


# ============================================================
# LOAD CHECKPOINT
# ============================================================
def load_model(checkpoint_dir: str, device: torch.device) -> MultiTaskCrossEncoderRoberta:
    print(f"\n[1/4] Loading checkpoint from  {checkpoint_dir}")

    # Rebuild model skeleton from the saved config
    model = MultiTaskCrossEncoderRoberta(
        model_name=checkpoint_dir,   # tokenizer/config live here
        num_labels=NUM_LABELS,
        dropout=0.0,                 # disable dropout for export
    )

    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {checkpoint_dir}")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("    ✓ Weights loaded")
    return model


# ============================================================
# EXPORT TO ONNX
# ============================================================
def export_onnx(
    model: MultiTaskCrossEncoderRoberta,
    tokenizer,
    output_path: str,
    max_length: int,
    device: torch.device,
    opset: int = 14,
):
    print(f"\n[2/4] Exporting to ONNX  →  {output_path}  (opset {opset})")

    wrapper = OnnxExportWrapper(model).to(device)
    wrapper.eval()

    # Dummy inputs for tracing
    dummy_input_ids = torch.ones(
        (1, max_length), dtype=torch.long, device=device
    )
    dummy_attention_mask = torch.ones(
        (1, max_length), dtype=torch.long, device=device
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    # Validate ONNX graph
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("    ✓ ONNX graph validated")
    print(f"    ✓ Saved  →  {output_path}")


# ============================================================
# ONNX RUNTIME INFERENCE
# ============================================================
def build_ort_session(onnx_path: str) -> ort.InferenceSession:
    print(f"\n[3/4] Building OnnxRuntime session from  {onnx_path}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=providers
    )
    print(f"    ✓ Providers : {session.get_providers()}")
    return session


def ort_predict(
    session: ort.InferenceSession,
    tokenizer,
    text_a: str,
    text_b: str,
    max_length: int,
) -> dict:
    """Run one (text_a, text_b) pair through the ONNX session."""
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
    logits = session.run(["logits"], ort_inputs)[0]   # [1, 18]

    results = {}
    for i, task in enumerate(TASK_NAMES):
        task_logits = logits[0, i * NUM_LABELS : (i + 1) * NUM_LABELS]
        probs       = softmax(task_logits)
        pred_class  = int(np.argmax(probs))
        confidence  = float(probs[pred_class])
        results[task] = {
            "predicted_class": pred_class,
            "confidence":      round(confidence, 4),
            "probabilities":   [round(float(p), 4) for p in probs],
        }
    return results


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# PYTORCH vs ONNX CONSISTENCY CHECK
# ============================================================
def consistency_check(
    pt_model: MultiTaskCrossEncoderRoberta,
    session: ort.InferenceSession,
    tokenizer,
    text_a: str,
    text_b: str,
    max_length: int,
    device: torch.device,
    atol: float = 1e-4,
):
    print("\n[4/4] Consistency check  (PyTorch logits vs ONNX logits)")

    enc = tokenizer(
        text_a, text_b,
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        pt_logits = pt_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).detach().cpu().numpy()

    ort_inputs = {
        "input_ids":      input_ids.cpu().numpy().astype(np.int64),
        "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
    }
    onnx_logits = session.run(["logits"], ort_inputs)[0]

    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    status   = "✓  PASS" if max_diff < atol else "✗  FAIL"
    print(f"    Max logit diff = {max_diff:.6f}   [{status}]")
    return max_diff < atol


# ============================================================
# DEMO INFERENCE — pretty-print
# ============================================================
def demo_inference(session, tokenizer, max_length):
    # A few synthetic (prompt, correction) pairs
    samples = [
        {
            "text_a": (
                "Prompt Level: 8 "
                "Prompt: Describe your daily routine. "
                "Response: Every morning I wakes up at seven and have breakfast."
            ),
            "text_b": (
                "Every morning I wake up at seven and have breakfast."
            ),
            "description": "B1 — minor agreement error",
        },
        {
            "text_a": (
                "Prompt Level: 14 "
                "Prompt: Discuss the impact of social media on society. "
                "Response: Social media have a profound effect on how people "
                "communicates and shares informations daily."
            ),
            "text_b": (
                "Social media has a profound effect on how people communicate "
                "and share information daily."
            ),
            "description": "C1 — subject-verb & plural errors",
        },
        {
            "text_a": (
                "Prompt Level: 5 "
                "Prompt: What is your favourite food? "
                "Response: I like pizza because is delicious and I eat it every week."
            ),
            "text_b": (
                "I like pizza because it is delicious and I eat it every week."
            ),
            "description": "A2 — missing pronoun",
        },
    ]

    print("\n" + "=" * 70)
    print("DEMO INFERENCE RESULTS")
    print("=" * 70)

    for s in samples:
        preds = ort_predict(session, tokenizer, s["text_a"], s["text_b"], max_length)
        print(f"\n  [{s['description']}]")
        for task, info in preds.items():
            bar = "█" * info["predicted_class"] + "░" * (NUM_LABELS - 1 - info["predicted_class"])
            print(
                f"    {task:<12}  class={info['predicted_class']}  "
                f"conf={info['confidence']:.3f}  [{bar}]"
            )
            print(f"               probs={info['probabilities']}")
    print()


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Export MultiTask RoBERTa → ONNX")
    p.add_argument(
        "--checkpoint",
        default="model_saved/roberta-large-multitask-multitask_cross_encoder_gte60",
        help="Path to saved checkpoint directory (must contain pytorch_model.bin + config.json)",
    )
    p.add_argument(
        "--output_dir",
        default="onnx_model",
        help="Directory where the .onnx file will be saved",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Sequence length used during training (check eval_meta.json)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for PT vs ONNX consistency check",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    onnx_path = os.path.join(args.output_dir, "multitask_roberta.onnx")

    # ── 1. Load PyTorch model ──────────────────────────────────────────────
    model     = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # ── 2. Export ──────────────────────────────────────────────────────────
    export_onnx(model, tokenizer, onnx_path, args.max_length, device, args.opset)

    # ── 2b. Quantization dynamique ─────────────────────────────────────────────
    quantized_path = os.path.join(args.output_dir, "multitask_roberta_quantized.onnx")
    print(f"\n[2b] Quantization dynamique  →  {quantized_path}")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        #op_types_to_quantize=["MatMul", "Attention"],  # exclut Gather
    )
    print(f"    ✓ Modèle quantifié  →  {quantized_path}")

    # ── 3. Build ORT session ───────────────────────────────────────────────
    session = build_ort_session(onnx_path)

    # ── 4. Consistency check ───────────────────────────────────────────────
    sample_a = (
        "Prompt Level: 8 "
        "Prompt: Describe your hometown. "
        "Response: My city is very beauty and have many park."
    )
    sample_b = "My city is very beautiful and has many parks."

    passed = consistency_check(
        model, session, tokenizer,
        sample_a, sample_b,
        args.max_length, device, args.atol,
    )

    # ── 5. Session quantifiée pour l'inférence demo ────────────────────────────
    session_quantized = build_ort_session(quantized_path)
    demo_inference(session_quantized, tokenizer, args.max_length)

    # ── 6. Artefact summary ───────────────────────────────────────────────
    onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print("=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"  ONNX model   : {onnx_path}")
    print(f"  File size    : {onnx_size_mb:.1f} MB")
    print(f"  Opset        : {args.opset}")
    print(f"  Max length   : {args.max_length}")
    print(f"  Consistency  : {'PASS ✓' if passed else 'FAIL ✗'}  (vs modèle ONNX non-quantifié)")
    print("=" * 70)

    quant_size_mb = os.path.getsize(quantized_path) / 1024 / 1024
    print(f"  ONNX quantifié : {quantized_path}")
    print(f"  File size      : {quant_size_mb:.1f} MB")

    if not passed:
        raise RuntimeError(
            "PyTorch and ONNX outputs diverge — check opset / dynamic ops."
        )


if __name__ == "__main__":
    main()