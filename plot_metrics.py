"""Plot train/eval loss and per-checkpoint accuracy + F1 curves.

Two modes:
  1. Loss curves  — reads TensorBoard event files (no re-evaluation needed)
  2. Accuracy/F1  — evaluates every saved step_XXX checkpoint, then plots

Usage (from GALLa root):
  # Loss curves only (fast)
  python plot_metrics.py --tb_dir output/clone/tb/qwen2.5-coder-1.5b

  # Accuracy + F1 across checkpoints (slow — runs full inference per checkpoint)
  python plot_metrics.py --tb_dir output/clone/tb/qwen2.5-coder-1.5b \\
      --eval_checkpoints output/clone/qwen2.5-coder-1.5b \\
      --base_model Qwen/Qwen2.5-Coder-1.5B \\
      --test_file  data_sample/clone_test.jsonl \\
      --device     cuda
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt


# ── TensorBoard loss curves ───────────────────────────────────────────────────

def read_tb_scalars(tb_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard package not found — skipping loss curves.")
        return {}

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {"steps": [e.step for e in events],
                     "values": [e.value for e in events]}
    return data


def plot_loss(tb_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    ax = axes[0]
    if "training_loss" in tb_data:
        d = tb_data["training_loss"]
        ax.plot(d["steps"], d["values"], label="train loss", color="steelblue")
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    if "valid_loss" in tb_data:
        d = tb_data["valid_loss"]
        ax.plot(d["steps"], d["values"], label="eval loss", color="orange")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Loss curve saved → {out_path}")
    plt.close()


# ── Per-checkpoint accuracy / F1 ─────────────────────────────────────────────

def find_checkpoints(ckpt_root):
    """Return sorted list of step_XXX checkpoint dirs."""
    dirs = glob.glob(os.path.join(ckpt_root, "step_*"))
    def step_num(p):
        try:
            return int(os.path.basename(p).split("_")[1])
        except (IndexError, ValueError):
            return -1
    dirs = [d for d in dirs if step_num(d) >= 0]
    return sorted(dirs, key=step_num)


def evaluate_checkpoint(ckpt_dir, base_model, test_file, device, max_samples=None):
    """Run full-model evaluation and return metrics dict."""
    # Check if already evaluated
    cached = os.path.join(ckpt_dir, "eval_results_full.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    # Lazy import to avoid loading torch/dgl until needed
    import subprocess, sys
    cmd = [
        sys.executable, "evaluate_clone_full.py",
        "--checkpoint",  ckpt_dir,
        "--base_model",  base_model,
        "--test_file",   test_file,
        "--device",      device,
    ]
    if max_samples:
        cmd += ["--max_samples", str(max_samples)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] eval failed for {ckpt_dir}:\n{result.stderr[-300:]}")
        return None

    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)
    return None


def plot_accuracy_f1(steps, accuracies, f1s, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(steps, accuracies, marker="o", color="steelblue", label="Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_title("Accuracy per Checkpoint")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, f1s, marker="o", color="orange", label="F1 Score")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_title("F1 Score per Checkpoint")
    ax.set_xlabel("Step")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Accuracy/F1 curve saved → {out_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_dir",           default=None,
                        help="TensorBoard log dir for loss curves")
    parser.add_argument("--eval_checkpoints", default=None,
                        help="Root dir containing step_XXX checkpoint folders")
    parser.add_argument("--base_model",       default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--test_file",        default="data_sample/clone_test.jsonl")
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--max_samples",      type=int, default=None,
                        help="Limit samples per checkpoint eval (faster)")
    parser.add_argument("--out_dir",          default="output/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Loss curves from TensorBoard ─────────────────────────────────────────
    if args.tb_dir:
        print(f"Reading TensorBoard logs from {args.tb_dir}")
        tb_data = read_tb_scalars(args.tb_dir)
        if tb_data:
            plot_loss(tb_data, os.path.join(args.out_dir, "loss_curves.png"))
        else:
            print("  No TensorBoard data found.")

    # ── Accuracy / F1 across checkpoints ─────────────────────────────────────
    if args.eval_checkpoints:
        ckpt_dirs = find_checkpoints(args.eval_checkpoints)
        if not ckpt_dirs:
            print(f"No step_XXX checkpoints found in {args.eval_checkpoints}")
        else:
            print(f"Found {len(ckpt_dirs)} checkpoints to evaluate")
            steps, accuracies, f1s = [], [], []
            for ckpt in ckpt_dirs:
                step = int(os.path.basename(ckpt).split("_")[1])
                print(f"  Evaluating step_{step} ...")
                metrics = evaluate_checkpoint(
                    ckpt, args.base_model, args.test_file, args.device, args.max_samples
                )
                if metrics:
                    steps.append(step)
                    accuracies.append(metrics["accuracy"])
                    f1s.append(metrics["f1"])
                    print(f"    accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

            if steps:
                plot_accuracy_f1(
                    steps, accuracies, f1s,
                    os.path.join(args.out_dir, "accuracy_f1_curve.png")
                )

                # Save combined summary CSV
                import csv
                csv_path = os.path.join(args.out_dir, "checkpoint_metrics.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["step", "accuracy", "f1",
                                                       "precision", "recall"])
                    w.writeheader()
                    for ckpt in ckpt_dirs:
                        step = int(os.path.basename(ckpt).split("_")[1])
                        cached = os.path.join(ckpt, "eval_results_full.json")
                        if os.path.exists(cached):
                            with open(cached) as cf:
                                m = json.load(cf)
                            w.writerow({"step": step, "accuracy": m["accuracy"],
                                        "f1": m["f1"], "precision": m["precision"],
                                        "recall": m["recall"]})
                print(f"Checkpoint metrics saved → {csv_path}")

    if not args.tb_dir and not args.eval_checkpoints:
        parser.print_help()


if __name__ == "__main__":
    main()
