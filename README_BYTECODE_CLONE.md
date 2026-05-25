# Cross-Language Code Clone Detection with GALLa + Bytecode2Vec

Detect whether a Java program and a Python program solve the same problem (cross-language
code clones) using the **GALLa** graph-aligned LLM pipeline on the **AtCoder Bytecode2Vec**
dataset.

---

## Overview

| Item | Detail |
|---|---|
| Task | Binary classification — clone (yes) vs non-clone (no) |
| Languages | Java ↔ Python (every pair is cross-language) |
| Graph | CFG + DFG of bytecode basic blocks; both files merged into one graph per pair |
| Node features | Pre-computed 200-dim Bytecode2Vec embeddings (mean-pooled per basic block) |
| Model | Qwen2.5-Coder-1.5B + LoRA + DUPLEX GNN + cross-attention adapter |
| Dataset | AtCoder pairs — 256k usable train pairs, 64k usable test pairs (1:5 pos/neg) |

---

## Repository Structure

```
mds-project2/
├── data/
│   ├── prepare_bytecode_clone.py   # NEW — converts Bytecode2Vec dataset → JSONL
│   ├── graph_dataset.py            # loads JSONL for training
│   └── preprocess_data.py          # tokenises + encodes graph samples
├── modeling/
│   ├── model.py                    # LM + GNN + Adapter (Model class)
│   ├── duplex.py                   # DUPLEX GNN (GAT-based)
│   └── gatconv.py                  # DGL GATConv layer
├── configs/
│   ├── config_bytecode_clone.json  # NEW — training config for this task (sample)
│   └── config_clone.json           # original BigCloneBench config (reference)
├── run.py                          # training entry point
├── evaluate_clone.py               # evaluation (LoRA weights only, text-only inference)
├── evaluate_clone_full.py          # evaluation (full GALLa: GNN + adapter + LM)
├── requirements.txt                # all dependencies
└── node_type_embedding.pth         # AST node embeddings (not used for bytecode task)
```

---

## Dataset Paths (server)

```
Graph JSONs (Java):  /Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/
                       Atcoder/atcoder/src/javaPyFlowInfoJson/JavaJson/
Graph JSONs (Python): /Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/
                       Atcoder/atcoder/src/javaPyFlowInfoJson/PyJson/
Train pairs CSV:     /Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/
                       shuffled_final_train_bytecode2vec.csv
Test pairs CSV:      /Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/
                       shuffled_final_test_bytecode2vec.csv
```

---

## Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> The `requirements.txt` is pinned for **Python 3.8** and **CUDA 12.x** (cu121 wheels,
> forward-compatible with CUDA 12.4 driver). If you use a different Python or CUDA version,
> adjust the index URLs and package versions accordingly.

---

## Step-by-Step Execution

### Step 1 — Prepare the data

#### Sample run (quick test — 1 000 train, 200 test pairs, balanced 1:1)

```bash
python data/prepare_bytecode_clone.py \
    --n_train 1000 \
    --n_test  200
```

Output files written to `data_sample/bytecode_clone/`:
```
data_sample/bytecode_clone/
├── train.jsonl   # 1 000 pairs
└── test.jsonl    #   200 pairs
```

Each line in the JSONL has:
```json
{
  "embeddings": [[...200 floats...], ...],   // one 200-dim vector per basic block
  "edge_index": [[0,1], [1,2], ...],         // CFG + DFG edges
  "question":   "Are the following Java and Python programs ...",
  "bot":        "yes",                       // ground-truth answer
  "label":      1                            // numeric label for evaluation scripts
}
```

#### Full run (all usable pairs — ~256k train, ~64k test)

```bash
python data/prepare_bytecode_clone.py \
    --n_train 250000 \
    --n_test  62000 \
    --out_dir data_sample/bytecode_clone_full
```

Then update `data_dir` in the config (or pass a custom config) to point to
`data_sample/bytecode_clone_full`.

> **Note:** The full dataset has a 1:5 positive/negative ratio. The data prep script
> draws balanced samples by default (1:1). To preserve the natural imbalance, edit
> `half_train`/`half_neg` in `load_pairs()` inside `prepare_bytecode_clone.py`.

---

### Step 2 — Train (sample)

```bash
python run.py --train_config configs/config_bytecode_clone.json
```

Key settings in `configs/config_bytecode_clone.json`:

| Parameter | Value | Notes |
|---|---|---|
| `graph_embedding_dim` | 200 | matches Bytecode2Vec output dimension |
| `graph_hidden_dim` | 512 | GNN hidden size |
| `graph_token_num` | 64 | number of soft graph tokens injected into LM |
| `mode` | ft | fine-tuning (not pre-training) |
| `lora` | true | only LoRA weights + GNN + adapter are trained |
| `num_train_epochs` | 15 | reduce to 5 for a quick smoke test |
| `data_split` | 90,10 | 90% of JSONL used for training, 10% for validation |

Checkpoints are saved to `output/bytecode_clone/qwen2.5-coder-1.5b/`.
The best checkpoint (lowest validation loss) is saved to `.../best_checkpoint/`.

#### Multi-GPU training with Accelerate

```bash
accelerate launch --config_file accelerate_ds_config.yaml \
    run.py --train_config configs/config_bytecode_clone.json
```

---

### Step 3 — Evaluate

Use `evaluate_clone.py` for the bytecode task. It loads the LoRA-adapted LM
and evaluates using text-only inference (the graph knowledge is baked into the
LoRA weights during fine-tuning).

```bash
python evaluate_clone.py \
    --checkpoint output/bytecode_clone/qwen2.5-coder-1.5b/best_checkpoint \
    --base_model Qwen/Qwen2.5-Coder-1.5B \
    --test_file  data_sample/bytecode_clone/test.jsonl \
    --device     cuda
```

Sample output:
```
==================================================
Clone Detection Evaluation Results
==================================================
  Samples evaluated : 200
  Accuracy          : 0.8350
  Precision         : 0.8100
  Recall            : 0.8700
  F1 Score          : 0.8389

Confusion Matrix (rows=actual, cols=predicted):
              Pred NO   Pred YES
  Actual NO      87        13
  Actual YES     13        87
==================================================
Results saved to output/.../best_checkpoint/eval_results.json
```

#### Limit evaluation to first N samples (quick sanity check)

```bash
python evaluate_clone.py \
    --checkpoint output/bytecode_clone/qwen2.5-coder-1.5b/best_checkpoint \
    --base_model Qwen/Qwen2.5-Coder-1.5B \
    --test_file  data_sample/bytecode_clone/test.jsonl \
    --max_samples 20 \
    --device cuda
```

> **Note on `evaluate_clone_full.py`:** This script runs the full GALLa inference
> pipeline (GNN → Adapter → LM) and is more accurate for the original AST-based
> clone task. For the bytecode task it requires adaptation: it currently expects
> `node_ids` (integer type codes) but our JSONL has pre-computed `embeddings`.
> Use `evaluate_clone.py` for now.

---

## Scaling to the Full Dataset

1. Run data prep with full counts (Step 1, full run command above).
2. Copy `configs/config_bytecode_clone.json` to `configs/config_bytecode_clone_full.json`
   and update:
   ```json
   {
     "data_dir": "data_sample/bytecode_clone_full",
     "output_dir": "output/bytecode_clone_full/qwen2.5-coder-1.5b",
     "tb_dir": "output/bytecode_clone_full/tb/qwen2.5-coder-1.5b",
     "num_train_epochs": 5,
     "gradient_accumulation_steps": 8
   }
   ```
3. Train:
   ```bash
   accelerate launch --config_file accelerate_ds_config.yaml \
       run.py --train_config configs/config_bytecode_clone_full.json
   ```
4. Evaluate on full test set:
   ```bash
   python evaluate_clone.py \
       --checkpoint output/bytecode_clone_full/qwen2.5-coder-1.5b/best_checkpoint \
       --base_model Qwen/Qwen2.5-Coder-1.5B \
       --test_file  data_sample/bytecode_clone_full/test.jsonl \
       --device     cuda
   ```

---

## How the Graph Is Built (Summary)

```
Java file JSON                    Python file JSON
    │                                   │
    ▼                                   ▼
For each method:                 For each method:
  For each basic block:            For each basic block:
    mean-pool opcode embeddings      mean-pool opcode embeddings
    → one 200-dim node vector        → one 200-dim node vector
  CFG + DFG edges                  CFG + DFG edges
    │                                   │
    └───────────── concatenate ─────────┘
                       │
          Python node IDs offset by #Java blocks
                       │
              Combined graph per pair
              (no cross-language edges)
                       │
                  DUPLEX GNN
              (3-layer GAT on DGL)
                       │
           Cross-attention Adapter
         (learnable queries → graph tokens)
                       │
         Injected into LM input embeddings
         at [graph_pad] token positions
                       │
              Qwen2.5-Coder-1.5B + LoRA
                       │
              Predict: "yes" or "no"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | Activate your virtualenv: `source venv/bin/activate` |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` |
| `dgl` install fails | Ensure CUDA toolkit matches; check [DGL install guide](https://www.dgl.ai/pages/start.html) |
| Data prep writes 0 samples | Check that the JSON paths are accessible and indices resolve to files |
| `KeyError: label` in evaluate | Re-run `prepare_bytecode_clone.py` — older outputs may not have the `label` field |
