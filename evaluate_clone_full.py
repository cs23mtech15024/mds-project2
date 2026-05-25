"""Evaluate a GALLa-fine-tuned model on the clone-detection test set.

Uses the FULL GALLa pipeline (GNN → Adapter → LoRA LM) for inference,
matching exactly what happened during training.  The text-only evaluate_clone.py
is unreliable because it never injects graph tokens, leaving the model to rely on
its LLM prior ("yes" bias).

Usage (from GALLa root):
  python evaluate_clone_full.py \\
      --checkpoint output/clone/qwen2.5-coder-1.5b/final_step_XXX \\
      --base_model  Qwen/Qwen2.5-Coder-1.5B \\
      --test_file   data_sample/clone_test.jsonl \\
      --device      cuda
"""

import argparse
import csv
import json
import os
import sys

import torch
import dgl
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling.duplex import DUPLEX
from modeling.model import Adapter

GRAPH_PAD_TOKEN = "<｜graph_pad｜>"
HUMAN_MARKER    = "<s>human\n"
BOT_MARKER      = "<s>bot\n"


# ── Minimal args container ─────────────────────────────────────────────────────

class _Args:
    pass


def _build_args(base_model, graph_token_num, graph_hidden_dim):
    a = _Args()
    a.pretrained_model_path = base_model
    a.model_type            = "qwen2.5"
    a.attn_implementation   = "eager"
    a.lora                  = True
    a.lora_rank             = 16
    a.lora_alpha            = 16
    a.graph_token_num       = graph_token_num
    a.graph_embedding_dim   = 256
    a.graph_hidden_dim      = graph_hidden_dim
    a.graph_node_types      = 43
    a.graph_pad_token       = GRAPH_PAD_TOKEN
    a.graph_pad_id          = None   # set after tokenizer is loaded
    a.num_heads             = 8
    a.lm_hidden_size        = None   # set after LM is loaded
    return a


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(base_model_path, checkpoint_dir, graph_token_num, graph_hidden_dim, device):
    args = _build_args(base_model_path, graph_token_num, graph_hidden_dim)

    # Tokenizer (saved to checkpoint dir, includes graph_pad_token)
    print(f"Loading tokenizer from {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    args.graph_pad_id = tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN)
    print(f"  graph_pad_id = {args.graph_pad_id}")

    # Base LM
    print(f"Loading base LM from {base_model_path}")
    base_lm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    base_lm.config.use_cache = True
    args.lm_hidden_size = base_lm.config.hidden_size
    print(f"  lm_hidden_size = {args.lm_hidden_size}")

    # LoRA adapter — load then merge into base weights for faster inference
    print(f"Loading LoRA adapter from {checkpoint_dir}")
    lm = PeftModel.from_pretrained(base_lm, checkpoint_dir)
    print("  Merging LoRA weights into base model...")
    lm = lm.merge_and_unload()   # fuses LoRA into base weights; no accuracy change
    lm.config.use_cache = True
    lm = lm.to(device)
    lm.eval()

    # GNN (DUPLEX)
    class _GNNArgs:
        dr_rate      = 0.0
        n_layers     = 3
        fusion_layer = 1
        input_dim    = 200  # bytecode embedding dim
        hidden_dim   = graph_hidden_dim
        output_dim   = graph_hidden_dim
        head         = 1
        fusion       = None

    print(f"Loading GNN from {checkpoint_dir}/GNN.pth")
    gnn = DUPLEX(_GNNArgs())
    gnn.load_state_dict(torch.load(f"{checkpoint_dir}/GNN.pth", weights_only=True, map_location="cpu"))
    gnn = gnn.to(device)
    gnn.eval()

    # Cross-attention Adapter
    print(f"Loading Adapter from {checkpoint_dir}/adapter.pth")
    adapter = Adapter(args)
    adapter.load_state_dict(torch.load(f"{checkpoint_dir}/adapter.pth", weights_only=True, map_location="cpu"))
    adapter = adapter.to(device)
    adapter.eval()

    return lm, gnn, adapter, tokenizer, args


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict(lm, gnn, adapter, tokenizer, sample, node_type_emb, args, device):
    """Full GALLa forward: GNN → Adapter → inject graph tokens → LM logits."""

    # 1. Build DGL graph from stored node/edge data
    edges = torch.LongTensor(sample["edge_index"]).t().contiguous()
    if "embeddings" in sample:
        # pre-computed bytecode embeddings (one 200-dim vector per basic block)
        num_nodes = len(sample["embeddings"])
        g     = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
        nfeats = torch.tensor(sample["embeddings"], dtype=torch.float32).to(device)
    else:
        # AST node type ids looked up in embedding table
        g     = dgl.graph((edges[0], edges[1]), num_nodes=len(sample["node_ids"]))
        nfeats = node_type_emb[torch.tensor(sample["node_ids"])].to(device)
    g = g.to(device)
    g.ndata["x"] = nfeats

    # 2. Build input token sequence (graph pads come before the question)
    human_ids    = args.human_ids
    graph_pads   = [args.graph_pad_id] * args.graph_token_num
    question_ids = tokenizer.encode(f"\n{sample['question']}\n", add_special_tokens=False)
    bot_ids      = args.bot_ids

    input_ids   = human_ids + graph_pads + question_ids + bot_ids
    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 3. Get LM token embeddings (merged model: plain AutoModelForCausalLM path)
    embed_fn    = lm.model.embed_tokens
    inputs_emb  = embed_fn(input_ids_t).clone()   # clone needed before in-place replace

    # 4. GNN forward
    gnn_dtype  = next(gnn.am_layers[0].parameters()).dtype
    g_batch    = dgl.batch([g])
    raw_feats  = g_batch.ndata["x"].to(gnn_dtype)
    gnn_out    = gnn(g_batch, raw_feats, raw_feats)  # (num_nodes, graph_hidden_dim)

    # 5. Adapter forward → (1, graph_token_num, lm_hidden_size)
    batch_num_nodes = torch.tensor([g.num_nodes()], device=device)
    graph_tokens    = adapter(gnn_out, batch_num_nodes)  # (1, 64, lm_hidden_size)

    # 6. Inject graph tokens at the placeholder positions
    positions = (input_ids_t[0] == args.graph_pad_id).nonzero().squeeze()
    pos_start = positions.min().item()
    pos_end   = positions.max().item()
    inputs_emb[0, pos_start : pos_end + 1] = graph_tokens[0].to(inputs_emb.dtype)

    # 7. Run LM (inputs_embeds skips the embedding look-up)
    outputs         = lm(inputs_embeds=inputs_emb, return_dict=True)
    next_tok_logits = outputs.logits[0, -1, :]

    yes_score = next_tok_logits[args.yes_id].item()
    no_score  = next_tok_logits[args.no_id].item()

    pred = 1 if yes_score > no_score else 0
    return pred, yes_score, no_score


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       default=None,
                        help="Fine-tuned checkpoint dir. Defaults to best_checkpoint inside --output_dir.")
    parser.add_argument("--base_model",       default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--test_file",        default="data_sample/bytecode_clone_100k/test.jsonl")
    parser.add_argument("--node_type_emb",    default="node_type_embedding.pth")
    parser.add_argument("--graph_token_num",  type=int, default=64)
    parser.add_argument("--graph_hidden_dim", type=int, default=512)
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir",       default="output/bytecode_clone_100k/qwen2.5-coder-1.5b",
                        help="Training output dir — used to locate best_checkpoint automatically.")
    parser.add_argument("--max_samples",      type=int, default=None)
    args = parser.parse_args()

    # Auto-resolve checkpoint to best_checkpoint if not explicitly provided
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, "best_checkpoint")
        print(f"No checkpoint specified — using best_checkpoint: {args.checkpoint}")

    lm, gnn, adapter, tokenizer, model_args = load_model(
        args.base_model, args.checkpoint,
        args.graph_token_num, args.graph_hidden_dim, args.device,
    )

    node_type_emb = torch.load(args.node_type_emb, weights_only=True).to(args.device)

    print(f"Loading test data from {args.test_file}")
    samples = []
    with open(args.test_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples")

    # Precompute token IDs and markers once (reused for every sample)
    model_args.yes_id    = tokenizer.encode("yes", add_special_tokens=False)[0]
    model_args.no_id     = tokenizer.encode("no",  add_special_tokens=False)[0]
    model_args.human_ids = tokenizer.encode(HUMAN_MARKER, add_special_tokens=False)
    model_args.bot_ids   = tokenizer.encode(BOT_MARKER,   add_special_tokens=False)

    # ── Resume: load already-evaluated samples ───────────────────────────────
    progress_path = f"{args.checkpoint}/eval_progress.json"
    per_sample_rows = []
    resume_from = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            per_sample_rows = json.load(f)
        resume_from = len(per_sample_rows)
        print(f"  Resuming from sample {resume_from} (found {resume_from} already evaluated)")

    y_true = [r["true_label"] for r in per_sample_rows]
    y_pred = [r["predicted"]  for r in per_sample_rows]

    for i, sample in enumerate(samples):
        if i < resume_from:
            continue
        pred, yes_score, no_score = predict(
            lm, gnn, adapter, tokenizer, sample, node_type_emb, model_args, args.device
        )
        y_true.append(sample["label"])
        y_pred.append(pred)
        per_sample_rows.append({
            "sample_id":   i + 1,
            "true_label":  sample["label"],
            "true_answer": sample["bot"],
            "predicted":   pred,
            "correct":     int(pred == sample["label"]),
            "yes_score":   round(yes_score, 4),
            "no_score":    round(no_score, 4),
            "score_gap":   round(yes_score - no_score, 4),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] yes={yes_score:.3f}  no={no_score:.3f}  "
                  f"pred={'yes' if pred else 'no '}  true={'yes' if sample['label'] else 'no '}")
        if (i + 1) % 500 == 0:
            with open(progress_path, "w") as f:
                json.dump(per_sample_rows, f)
            print(f"  [checkpoint saved at {i+1}]")

    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 55)
    print("Clone Detection Evaluation  (Full GALLa Model)")
    print("=" * 55)
    print(f"  Samples evaluated : {len(samples)}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print()
    print("  Confusion Matrix (rows = actual, cols = predicted):")
    print("                Pred NO   Pred YES")
    print(f"  Actual NO     {metrics['tn']:6d}    {metrics['fp']:6d}")
    print(f"  Actual YES    {metrics['fn']:6d}    {metrics['tp']:6d}")
    print("=" * 55)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_path = f"{args.checkpoint}/eval_results_full.json"
    with open(summary_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "test_file":  args.test_file,
            "n_samples":  len(samples),
            **metrics,
        }, f, indent=2)
    print(f"\nSummary  saved → {summary_path}")

    # ── Save per-sample CSV ───────────────────────────────────────────────────
    csv_path = f"{args.checkpoint}/eval_per_sample.csv"
    fieldnames = ["sample_id", "true_label", "true_answer", "predicted",
                  "correct", "yes_score", "no_score", "score_gap"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Per-sample saved → {csv_path}")

    # ── Save per-sample JSON ──────────────────────────────────────────────────
    detail_path = f"{args.checkpoint}/eval_per_sample.json"
    with open(detail_path, "w") as f:
        json.dump(per_sample_rows, f, indent=2)
    print(f"Per-sample saved → {detail_path}")

    # Remove progress file now that final results are saved
    if os.path.exists(progress_path):
        os.remove(progress_path)


if __name__ == "__main__":
    main()
