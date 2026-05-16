"""Evaluate a GALLa-fine-tuned model on the clone-detection test set.

Loads the LoRA-adapted LLM saved by train_utils.py, runs text-only inference
(the graph knowledge is baked into the LoRA weights), and reports:
  - Accuracy
  - Precision, Recall, F1  (binary, positive class = "clone")
  - Confusion matrix

Usage (from GALLa root):
  python evaluate_clone.py \
      --checkpoint output/clone/qwen2.5-coder-1.5b/final_step_XXX \
      --base_model  Qwen/Qwen2.5-Coder-1.5B \
      --test_file   data_sample/clone_test.jsonl \
      --device      cuda
"""

import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Token markers used during training (must match preprocess_data.py) ─────────
HUMAN_MARKER = "<s>human\n"
BOT_MARKER   = "<s>bot\n"


def build_prompt(question: str) -> str:
    """Format a question the same way the model was trained on."""
    return f"{HUMAN_MARKER}{question}\n{BOT_MARKER}"


def load_test_data(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append({
                "question": d["question"],
                "label":    d["label"],          # 1 = clone, 0 = not clone
                "bot":      d["bot"],             # ground-truth answer ("yes"/"no")
            })
    return samples


@torch.inference_mode()
def predict(model, tokenizer, prompt: str, device: str) -> tuple:
    """Compare logits for 'yes' vs 'no' at the next token position.

    Returns (predicted_label, yes_score, no_score).
    This is more reliable than greedy decoding for binary classification.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Logits for the very next token after the prompt
    next_token_logits = outputs.logits[0, -1, :]

    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids  = tokenizer.encode("no",  add_special_tokens=False)

    yes_score = next_token_logits[yes_ids[0]].item()
    no_score  = next_token_logits[no_ids[0]].item()

    pred = 1 if yes_score > no_score else 0
    return pred, yes_score, no_score


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

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to the GALLa fine-tuned checkpoint directory "
                             "(contains adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--base_model",  default="Qwen/Qwen2.5-Coder-1.5B",
                        help="HuggingFace base model name or local path")
    parser.add_argument("--test_file",   default="data_sample/clone_test.jsonl",
                        help="JSONL test set produced by data/prepare_clone_data.py")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to first N samples (useful for quick checks)")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model = model.to(args.device)
    model.eval()

    print(f"Loading test data from {args.test_file}")
    samples = load_test_data(args.test_file)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples to evaluate")

    y_true, y_pred = [], []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample["question"])
        pred, yes_score, no_score = predict(model, tokenizer, prompt, args.device)

        y_true.append(sample["label"])
        y_pred.append(pred)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] yes={yes_score:.3f} no={no_score:.3f} pred={pred} true={sample['label']}")

    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 50)
    print("Clone Detection Evaluation Results")
    print("=" * 50)
    print(f"  Samples evaluated : {len(samples)}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print()
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred NO   Pred YES")
    print(f"  Actual NO   {metrics['tn']:6d}    {metrics['fp']:6d}")
    print(f"  Actual YES  {metrics['fn']:6d}    {metrics['tp']:6d}")
    print("=" * 50)

    # Save results to file next to the checkpoint
    results_path = f"{args.checkpoint}/eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "test_file":  args.test_file,
            "n_samples":  len(samples),
            **metrics,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
