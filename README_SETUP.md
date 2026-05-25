# Cross-Language Code Clone Detection — Setup & Execution Guide

This guide covers the **complete setup from scratch** on a Linux server (Ubuntu) with CUDA 12.x,
including all known pitfalls and their fixes.

---

## Prerequisites

- Linux server with NVIDIA GPU (CUDA 12.x driver)
- ~15 GB free disk space in your home directory
- Access to the dataset at `/Pramana/ByteCode2Vec/`

---

## Step 1 — Install Miniconda (Python 3.10)

The codebase requires **Python 3.10**. Use Miniconda to avoid system Python conflicts.

```bash
# Download installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P ~/

# Install (accept defaults, say yes to conda init)
bash ~/Miniconda3-latest-Linux-x86_64.sh

# Accept Anaconda Terms of Service (required once)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Reload shell so conda is available
source ~/.bashrc
```

> **If `conda` is not found after reopening the terminal**, run:
> ```bash
> source ~/miniconda3/etc/profile.d/conda.sh
> ```
> Add this line to `~/.bashrc` to make it permanent.

---

## Step 2 — Create Conda Environment

```bash
conda create -n mds_env_v2 python=3.10 -y
conda activate mds_env_v2
```

---

## Step 3 — Install Dependencies

Disk space note: pip uses `/tmp` by default. If the root partition is small, redirect to home:

```bash
mkdir -p ~/pip_tmp ~/pip_cache
TMPDIR=~/pip_tmp pip install --cache-dir ~/pip_cache -r requirements.txt
```

> This installs PyTorch 2.3, DGL 2.3 (CUDA 12.1 wheels), Transformers, PEFT, Accelerate, DeepSpeed, and other dependencies.
> The install takes **10-15 minutes** on first run.

---

## Step 4 — Fix CUDA Library Path (required every session)

DGL requires `libnvrtc.so.12` which may not be on the default library path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Add permanently to avoid doing this every session:**
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

> If your CUDA is installed at a different path, find it with: `find /usr/local -name "libnvrtc.so*" 2>/dev/null`

---

## Step 5 — Clone / Navigate to Project

```bash
cd ~/mds-project/mds-project2
```

---

## Step 6 — Prepare the Dataset

The raw dataset lives at `/Pramana/ByteCode2Vec/`. The preparation script reads Java and Python
graph JSON files, builds combined graphs per pair, and writes JSONL files for training.

### Quick smoke test (5k pairs — ~5 minutes to prepare)

```bash
python data/prepare_bytecode_clone.py \
    --n_train 5000 \
    --n_test  1000 \
    --out_dir data_sample/bytecode_clone_5k
```

### Recommended run (50k pairs — ~30 minutes to prepare)

```bash
python data/prepare_bytecode_clone.py \
    --n_train 50000 \
    --n_test  10000 \
    --out_dir data_sample/bytecode_clone_50k
```

**Expected output:**
```
Building index maps...
  Java files: 15658, Py files: 23787
Sampling 5000 train pairs (2500 pos + 2500 neg)...
  Loaded 5000 pairs
Writing data_sample/bytecode_clone_5k/train.jsonl...
  wrote 3972, skipped 1028
Writing data_sample/bytecode_clone_5k/test.jsonl...
  wrote 794, skipped 206
```

Skipped pairs are ones where the graph JSON file is missing on disk — this is expected.

> **Dataset paths (hardcoded in the script):**
> - Java graphs: `/Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/Atcoder/atcoder/src/javaPyFlowInfoJson/JavaJson/`
> - Python graphs: `/Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/Atcoder/atcoder/src/javaPyFlowInfoJson/PyJson/`
> - Train pairs CSV: `/Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/shuffled_final_train_bytecode2vec.csv`
> - Test pairs CSV: `/Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/shuffled_final_test_bytecode2vec.csv`

---

## Step 7 — Update Config (if changing dataset size)

Edit `configs/config_bytecode_clone.json` and update `data_dir`:

```json
{
  "data_dir": "data_sample/bytecode_clone_50k",
  "output_dir": "output/bytecode_clone/qwen2.5-coder-1.5b",
  ...
}
```

The default config already points to `data_sample/bytecode_clone_5k` for the smoke test.

---

## Step 8 — Train

```bash
python run.py --train_config configs/config_bytecode_clone.json
```

**For multi-GPU training:**
```bash
accelerate launch --config_file accelerate_ds_config.yaml \
    run.py --train_config configs/config_bytecode_clone.json
```

**Expected training output:**
```
step 10: loss = 2.66
step 20: loss = 2.10
...
step 200: eval_loss = 1.45  ← checkpoint saved
...
Best checkpoint saved to output/bytecode_clone/qwen2.5-coder-1.5b/best_checkpoint
```

Checkpoints are saved every 200 steps. The best checkpoint (lowest validation loss) is saved to `best_checkpoint/`.

**Estimated training time:**
| Dataset | Epochs | Time |
|---------|--------|------|
| 5k pairs | 5 | ~1-2 hrs |
| 50k pairs | 5 | ~8-12 hrs |

---

## Step 9 — Evaluate

```bash
python evaluate_clone_full.py \
    --checkpoint output/bytecode_clone/qwen2.5-coder-1.5b/best_checkpoint \
    --base_model Qwen/Qwen2.5-Coder-1.5B \
    --test_file  data_sample/bytecode_clone_5k/test.jsonl \
    --device     cuda
```

**Expected output:**
```
=======================================================
Clone Detection Evaluation  (Full GALLa Model)
=======================================================
  Samples evaluated : 794
  Accuracy          : 0.8500
  Precision         : 0.8300
  Recall            : 0.8700
  F1 Score          : 0.8495

  Confusion Matrix (rows = actual, cols = predicted):
                Pred NO   Pred YES
  Actual NO        342        55
  Actual YES        63       334
=======================================================
```

Results are saved to:
- `best_checkpoint/eval_results_full.json` — summary metrics
- `best_checkpoint/eval_per_sample.csv` — per-sample predictions
- `best_checkpoint/eval_per_sample.json` — per-sample predictions (JSON)

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `conda: command not found` | `source ~/miniconda3/etc/profile.d/conda.sh` |
| `libnvrtc.so.12: cannot open shared object file` | `export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH` |
| `No space left on device` during pip install | `TMPDIR=~/pip_tmp pip install --cache-dir ~/pip_cache -r requirements.txt` |
| `ValueError: No objects to concatenate` in training | Re-check that JSONL files have `embeddings` field (re-run data prep) |
| `KeyError: label` in evaluate | Re-run `prepare_bytecode_clone.py` — old JSONL may lack the `label` field |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 in config |
| Data prep takes very long | `/Pramana/` is a network mount — 5k pairs takes ~10 min, 50k takes ~30 min |
| `evaluate_clone.py` gives 50% accuracy (all "no") | Use `evaluate_clone_full.py` instead — text-only inference doesn't inject graph tokens |

---

## Key Config Parameters

`configs/config_bytecode_clone.json`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_dir` | `data_sample/bytecode_clone_5k` | Change for larger runs |
| `graph_embedding_dim` | 200 | Must match Bytecode2Vec output dim |
| `graph_hidden_dim` | 512 | GNN hidden size |
| `graph_token_num` | 64 | Graph tokens injected into LM |
| `num_train_epochs` | 5 | Increase for larger datasets |
| `lora` | true | LoRA keeps memory usage low |
| `early_stopping` | true | Stops if val loss stalls for 5 evals |
| `data_split` | `90,10` | 90% train, 10% validation |

---

## Disk Space Budget

| Item | Size |
|------|------|
| Miniconda + env | ~8 GB |
| Hugging Face model cache (`~/.cache/huggingface`) | ~6 GB |
| Dataset JSONL (5k pairs) | ~120 MB |
| Dataset JSONL (50k pairs) | ~1.2 GB |
| Each checkpoint | ~120 MB |
| **Total (smoke test)** | **~15 GB** |
| **Total (50k run)** | **~16 GB** |

> User quota on this server is **50 GB**. Keep old checkpoints deleted after confirming results.

---

## Full Session Startup Checklist

Every time you open a new terminal:

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # if conda not found
conda activate mds_env_v2
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
cd ~/mds-project/mds-project2
```

> Add the `export LD_LIBRARY_PATH` line to `~/.bashrc` to skip it in future sessions.
