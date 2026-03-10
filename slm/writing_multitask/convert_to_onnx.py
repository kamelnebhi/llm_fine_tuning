"""
convert_to_onnx.py
------------------
Converts a trained MultiTaskCrossEncoderRoberta (v2) checkpoint to ONNX,
then runs a quick inference test to validate the export.

Usage:
    python convert_to_onnx.py \
        --checkpoint model_saved/roberta-large-multitask-multitask_cross_encoder_v4/checkpoint-XXX \
        --output_dir onnx_model \
        --max_length_acc 512 \
        --max_length_coh_rng 256
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from transformers import AutoTokenizer, RobertaModel
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

# ── constants (must match training script) ─────────────────────────────────────
NUM_LABELS = 6
TASK_NAMES = ["accuracy", "coherence", "range"]

def export_onnx_fixed(
    model, tokenizer, output_path,
    max_length_acc, max_length_coh_rng,
    device, opset=14,
):
    """Export with fixed shapes — needed for quantization on older onnxruntime."""
    print(f"    Exporting fixed-shape ONNX  →  {output_path}")

    wrapper = OnnxExportWrapper(model).to(device)
    wrapper.eval()

    dummy_ids_acc      = torch.ones((1, max_length_acc),     dtype=torch.long, device=device)
    dummy_mask_acc     = torch.ones((1, max_length_acc),     dtype=torch.long, device=device)
    dummy_ids_coh_rng  = torch.ones((1, max_length_coh_rng), dtype=torch.long, device=device)
    dummy_mask_coh_rng = torch.ones((1, max_length_coh_rng), dtype=torch.long, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids_acc, dummy_mask_acc, dummy_ids_coh_rng, dummy_mask_coh_rng),
            output_path,
            opset_version=opset,
            input_names=[
                "input_ids_acc", "attention_mask_acc",
                "input_ids_coh_rng", "attention_mask_coh_rng",
            ],
            output_names=["logits"],
            # No dynamic_axes — fixed shapes
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(output_path))
    print(f"    ✓ Fixed-shape export done")

def preprocess_gemm_to_matmul(input_path: str, output_path: str):
    """Convert Gemm nodes to MatMul+Add to fix quantization."""
    import onnx
    from onnx import numpy_helper
    import numpy as np

    model = onnx.load(input_path)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type != "Gemm":
            continue

        attrs = {a.name: a for a in node.attribute}
        transB = attrs.get("transB", None)
        transB = transB.i if transB else 0

        A = node.input[0]
        B = node.input[1]
        C = node.input[2] if len(node.input) > 2 else None
        Y = node.output[0]

        if transB and B in initializers:
            w = numpy_helper.to_array(initializers[B])
            w_t = w.T.copy()
            new_name = B + "_transposed"
            new_init = numpy_helper.from_array(w_t, name=new_name)
            graph.initializer.append(new_init)
            B_use = new_name
        else:
            B_use = B

        matmul_out = Y + "_matmul_out" if C else Y
        matmul_node = onnx.helper.make_node(
            "MatMul",
            inputs=[A, B_use],
            outputs=[matmul_out],
            name=node.name + "_matmul",
        )
        nodes_to_add.append(matmul_node)

        if C:
            add_node = onnx.helper.make_node(
                "Add",
                inputs=[matmul_out, C],
                outputs=[Y],
                name=node.name + "_add",
            )
            nodes_to_add.append(add_node)

        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(nodes_to_add)

    # Run shape inference to fill in missing type info
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)

    onnx.save(model, output_path)
    print(f"    ✓ Converted {len(nodes_to_remove)} Gemm → MatMul+Add  →  {output_path}")
    return len(nodes_to_remove)


# ============================================================
# ATTENTION POOLING  (from training script)
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


# ============================================================
# TASK ADAPTER  (from training script)
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
# MODEL DEFINITION  (exact copy from training script)
# ============================================================
class MultiTaskCrossEncoderRoberta(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.num_labels = num_labels
        trunk_dim       = 768

        # Attention pooling
        self.attn_pool = AttentionPooling(hidden_size)

        # Shared trunk (CLS + attn → 2*hidden)
        self.shared_trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
        )

        # Per-task adapters
        self.adapter_accuracy  = TaskAdapter(trunk_dim, bottleneck=64)
        self.adapter_coherence = TaskAdapter(trunk_dim, bottleneck=96)
        self.adapter_range     = TaskAdapter(trunk_dim, bottleneck=80)

        # Asymmetric projections
        proj_dim_acc = 256
        proj_dim_coh = 384
        proj_dim_rng = 320

        self.proj_accuracy  = self._make_proj(trunk_dim, proj_dim_acc, dropout)
        self.proj_coherence = self._make_proj(trunk_dim, proj_dim_coh, dropout)
        self.proj_range     = self._make_proj(trunk_dim, proj_dim_rng, dropout)

        # Classification heads
        self.head_accuracy  = nn.Linear(proj_dim_acc, num_labels)
        self.head_coherence = nn.Linear(proj_dim_coh, num_labels)
        self.head_range     = nn.Linear(proj_dim_rng, num_labels)

        # Uncertainty weighting (not used at inference, but needed to load weights)
        self.log_sigma = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _make_proj(in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def _encode_and_pool(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr  = out.last_hidden_state[:, 0, :]
        attn_repr = self.attn_pool(out.last_hidden_state, attention_mask)
        combined  = torch.cat([cls_repr, attn_repr], dim=-1)
        return self.shared_trunk(combined)

    def forward(
        self,
        input_ids_acc,
        attention_mask_acc,
        input_ids_coh_rng,
        attention_mask_coh_rng,
        **kwargs,
    ):
        # ── Pass 1: accuracy (text_a + correction) ───────────────
        shared_acc = self._encode_and_pool(input_ids_acc, attention_mask_acc)
        feat_acc   = self.proj_accuracy(self.adapter_accuracy(shared_acc))
        logits_acc = self.head_accuracy(feat_acc)

        # ── Pass 2: coherence + range (text_a only) ──────────────
        shared_coh_rng = self._encode_and_pool(input_ids_coh_rng, attention_mask_coh_rng)
        feat_coh       = self.proj_coherence(self.adapter_coherence(shared_coh_rng))
        feat_rng       = self.proj_range(self.adapter_range(shared_coh_rng))
        logits_coh     = self.head_coherence(feat_coh)
        logits_rng     = self.head_range(feat_rng)

        logits = torch.cat([logits_acc, logits_coh, logits_rng], dim=-1)
        return logits   # [batch, 18] — plain tensor for ONNX


# ============================================================
# ONNX WRAPPER — 4 inputs → 1 output
# ============================================================
class OnnxExportWrapper(nn.Module):
    """Wrapper accepting 4 tensors for torch.onnx.export."""

    def __init__(self, model: MultiTaskCrossEncoderRoberta):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids_acc:        torch.Tensor,
        attention_mask_acc:   torch.Tensor,
        input_ids_coh_rng:    torch.Tensor,
        attention_mask_coh_rng: torch.Tensor,
    ):
        return self.model(
            input_ids_acc=input_ids_acc,
            attention_mask_acc=attention_mask_acc,
            input_ids_coh_rng=input_ids_coh_rng,
            attention_mask_coh_rng=attention_mask_coh_rng,
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================
def load_model(checkpoint_dir: str, device: torch.device) -> MultiTaskCrossEncoderRoberta:
    print(f"\n[1/4] Loading checkpoint from  {checkpoint_dir}")

    model = MultiTaskCrossEncoderRoberta(
        model_name=checkpoint_dir,
        num_labels=NUM_LABELS,
        dropout=0.0,          # disable dropout for export
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
# EXPORT TO ONNX — 4 inputs
# ============================================================
def export_onnx(
    model: MultiTaskCrossEncoderRoberta,
    tokenizer,
    output_path: str,
    max_length_acc: int,
    max_length_coh_rng: int,
    device: torch.device,
    opset: int = 14,
):
    print(f"\n[2/4] Exporting to ONNX  →  {output_path}  (opset {opset})")

    wrapper = OnnxExportWrapper(model).to(device)
    wrapper.eval()

    # Dummy inputs — two different sequence lengths
    dummy_ids_acc      = torch.ones((1, max_length_acc),     dtype=torch.long, device=device)
    dummy_mask_acc     = torch.ones((1, max_length_acc),     dtype=torch.long, device=device)
    dummy_ids_coh_rng  = torch.ones((1, max_length_coh_rng), dtype=torch.long, device=device)
    dummy_mask_coh_rng = torch.ones((1, max_length_coh_rng), dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids_acc, dummy_mask_acc, dummy_ids_coh_rng, dummy_mask_coh_rng),
            output_path,
            opset_version=opset,
            input_names=[
                "input_ids_acc",
                "attention_mask_acc",
                "input_ids_coh_rng",
                "attention_mask_coh_rng",
            ],
            output_names=["logits"],
            dynamic_axes={
                "input_ids_acc":          {0: "batch_size", 1: "seq_len_acc"},
                "attention_mask_acc":     {0: "batch_size", 1: "seq_len_acc"},
                "input_ids_coh_rng":      {0: "batch_size", 1: "seq_len_coh_rng"},
                "attention_mask_coh_rng": {0: "batch_size", 1: "seq_len_coh_rng"},
                "logits":                 {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("    ✓ ONNX graph validated")
    print(f"    ✓ Saved  →  {output_path}")


# ============================================================
# ONNX RUNTIME SESSION
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


# ============================================================
# TOKENIZATION HELPERS
# ============================================================
def tokenize_for_onnx(
    tokenizer,
    text_a: str,
    text_b: str,
    max_length_acc: int,
    max_length_coh_rng: int,
) -> dict:
    """Tokenize one sample into the 4 numpy arrays needed by the ONNX model."""

    # Pass 1: accuracy  →  text_a + text_b  (cross-encoder pair)
    enc_acc = tokenizer(
        text_a,
        text_b,
        truncation=True,
        padding="max_length",
        max_length=max_length_acc,
        return_tensors="np",
    )

    # Pass 2: coherence/range  →  text_a only
    enc_coh_rng = tokenizer(
        text_a,
        truncation=True,
        padding="max_length",
        max_length=max_length_coh_rng,
        return_tensors="np",
    )

    return {
        "input_ids_acc":          enc_acc["input_ids"].astype(np.int64),
        "attention_mask_acc":     enc_acc["attention_mask"].astype(np.int64),
        "input_ids_coh_rng":      enc_coh_rng["input_ids"].astype(np.int64),
        "attention_mask_coh_rng": enc_coh_rng["attention_mask"].astype(np.int64),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# ONNX RUNTIME INFERENCE
# ============================================================
def ort_predict(
    session: ort.InferenceSession,
    tokenizer,
    text_a: str,
    text_b: str,
    max_length_acc: int,
    max_length_coh_rng: int,
) -> dict:
    """Run one (text_a, text_b) pair through the ONNX session."""

    ort_inputs = tokenize_for_onnx(
        tokenizer, text_a, text_b, max_length_acc, max_length_coh_rng
    )
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


# ============================================================
# PYTORCH vs ONNX CONSISTENCY CHECK
# ============================================================
def consistency_check(
    pt_model: MultiTaskCrossEncoderRoberta,
    session: ort.InferenceSession,
    tokenizer,
    text_a: str,
    text_b: str,
    max_length_acc: int,
    max_length_coh_rng: int,
    device: torch.device,
    atol: float = 1e-4,
):
    print("\n[4/4] Consistency check  (PyTorch logits vs ONNX logits)")

    # ── PyTorch inference ──────────────────────────────────────────────
    enc_acc = tokenizer(
        text_a, text_b,
        truncation=True, padding="max_length",
        max_length=max_length_acc, return_tensors="pt",
    )
    enc_coh_rng = tokenizer(
        text_a,
        truncation=True, padding="max_length",
        max_length=max_length_coh_rng, return_tensors="pt",
    )

    with torch.no_grad():
        pt_logits = pt_model(
            input_ids_acc=enc_acc["input_ids"].to(device),
            attention_mask_acc=enc_acc["attention_mask"].to(device),
            input_ids_coh_rng=enc_coh_rng["input_ids"].to(device),
            attention_mask_coh_rng=enc_coh_rng["attention_mask"].to(device),
        ).detach().cpu().numpy()

    # ── ONNX inference ─────────────────────────────────────────────────
    ort_inputs = tokenize_for_onnx(
        tokenizer, text_a, text_b, max_length_acc, max_length_coh_rng
    )
    onnx_logits = session.run(["logits"], ort_inputs)[0]

    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    status   = "✓  PASS" if max_diff < atol else "✗  FAIL"
    print(f"    Max logit diff = {max_diff:.6f}   [{status}]")

    # ── Per-task breakdown ─────────────────────────────────────────────
    for i, task in enumerate(TASK_NAMES):
        sl = slice(i * NUM_LABELS, (i + 1) * NUM_LABELS)
        task_diff = float(np.abs(pt_logits[:, sl] - onnx_logits[:, sl]).max())
        pt_pred   = int(np.argmax(pt_logits[:, sl], axis=-1)[0])
        onnx_pred = int(np.argmax(onnx_logits[:, sl], axis=-1)[0])
        match     = "✓" if pt_pred == onnx_pred else "✗"
        print(
            f"    {task:<12}  max_diff={task_diff:.6f}  "
            f"pt_pred={pt_pred}  onnx_pred={onnx_pred}  {match}"
        )

    return max_diff < atol


# ============================================================
# DEMO INFERENCE
# ============================================================
def demo_inference(session, tokenizer, max_length_acc, max_length_coh_rng):
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
    print(f"  max_length_acc={max_length_acc}  max_length_coh_rng={max_length_coh_rng}")

    for s in samples:
        preds = ort_predict(
            session, tokenizer,
            s["text_a"], s["text_b"],
            max_length_acc, max_length_coh_rng,
        )
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
    p = argparse.ArgumentParser(description="Export MultiTask RoBERTa v2 → ONNX")
    p.add_argument(
        "--checkpoint",
        default="model_saved/roberta-large-multitask-multitask_cross_encoder_v4",
        help="Path to saved checkpoint directory",
    )
    p.add_argument("--output_dir", default="onnx_model")
    p.add_argument("--max_length_acc",     type=int, default=512,
                   help="Max sequence length for accuracy (text_a + correction)")
    p.add_argument("--max_length_coh_rng", type=int, default=256,
                   help="Max sequence length for coherence/range (text_a only)")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--atol",  type=float, default=1e-2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    onnx_path = os.path.join(args.output_dir, "multitask_roberta.onnx")

    # ── 1. Load PyTorch model ──────────────────────────────────────────
    model     = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # ── 2. Export ──────────────────────────────────────────────────────
    export_onnx(
        model, tokenizer, onnx_path,
        args.max_length_acc, args.max_length_coh_rng,
        device, args.opset,
    )

    # ── 2b. Quantization skip─────────────────────────────────────────────
    quantized_path = onnx_path
    has_quantized  = False
    print(f"\n[2b] Quantization skipped — full-precision ONNX")

    # ── 3. Build ORT session ──────────────────────────────────────────
    session = build_ort_session(onnx_path)

    # ── 4. Consistency check ──────────────────────────────────────────
    sample_a = (
        "Prompt Level: 8 "
        "Prompt: Describe your hometown. "
        "Response: My city is very beauty and have many park."
    )
    sample_b = "My city is very beautiful and has many parks."

    passed = consistency_check(
        model, session, tokenizer,
        sample_a, sample_b,
        args.max_length_acc, args.max_length_coh_rng,
        device, args.atol,
    )

    # ── 5. Demo on quantized model ────────────────────────────────────
    session_quantized = build_ort_session(quantized_path)
    demo_inference(
        session_quantized, tokenizer,
        args.max_length_acc, args.max_length_coh_rng,
    )

    # ── 6. Summary ────────────────────────────────────────────────────
    onnx_size_mb  = os.path.getsize(onnx_path) / 1024 / 1024
    quant_size_mb = os.path.getsize(quantized_path) / 1024 / 1024

    print("=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"  ONNX model       : {onnx_path}  ({onnx_size_mb:.1f} MB)")
    #print(f"  ONNX quantized   : {quantized_path}  ({quant_size_mb:.1f} MB)")
    print(f"  Compression      : {onnx_size_mb / quant_size_mb:.1f}x")
    print(f"  Opset            : {args.opset}")
    print(f"  max_length_acc   : {args.max_length_acc}")
    print(f"  max_length_coh   : {args.max_length_coh_rng}")
    print(f"  Consistency      : {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 70)

    if not passed:
        raise RuntimeError(
            "PyTorch and ONNX outputs diverge — check opset / dynamic ops."
        )


if __name__ == "__main__":
    main()