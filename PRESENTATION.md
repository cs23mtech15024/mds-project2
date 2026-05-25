# Cross-Language Code Clone Detection using GALLa + Bytecode2Vec

**Master's Project Presentation**

---

## 1. Problem Statement

**Code Clone Detection** — identifying whether two programs implement the same functionality.

This project focuses on the **cross-language** variant:
- Given a **Java** program and a **Python** program
- Determine if they solve the **same problem** (clone) or not (non-clone)
- Binary classification: **yes** (clone) / **no** (non-clone)

### Why is this hard?
- Different syntax, idioms, and language semantics
- No token-level overlap between Java and Python code
- Cannot use text similarity — must understand program semantics
- Requires reasoning over **program structure**, not just surface code

---

## 2. Dataset

### Source
**AtCoder Bytecode2Vec Dataset** — competitive programming submissions (AtCoder judge)

| Property | Detail |
|---|---|
| Languages | Java ↔ Python (all pairs are cross-language) |
| Pairs in dataset | ~338k train, ~84k test (raw CSV) |
| Usable pairs | ~256k train, ~64k test (after filtering missing graphs) |
| Class ratio | 1:5 positive:negative (natural) |
| Our experiment | Balanced 1:1 sampling |

### Graph Representation
Each program is represented as a **Control Flow Graph (CFG) + Data Flow Graph (DFG)** of bytecode:

```
Java source code
       │
       ▼
  JVM Bytecode
       │
       ▼
  Basic Blocks (nodes)
  ┌─────────────────────┐
  │ opcodes per block   │  → mean-pool opcode embeddings
  │ ILOAD, IADD, ISTORE │  → one 200-dim vector per block
  └─────────────────────┘
       │
   CFG edges (control flow)
   DFG edges (data flow)
```

### Node Features
- Each basic block → **200-dim Bytecode2Vec embedding** (mean-pooled opcode embeddings)
- Pre-computed, not learned during training

### Combined Graph per Pair
```
Java graph (N_j nodes)    Python graph (N_p nodes)
        │                          │
        └────────── concat ─────────┘
                       │
          Python node IDs offset by N_j
          (no cross-language edges added)
                       │
          Single combined graph per pair
```

### Data Splits Across Experiments

| Experiment | Train (80%) | Val (20%) | Test (held-out) | Total Pairs |
|------------|-------------|-----------|-----------------|-------------|
| 5k smoke test | ~3,575 | ~397 | 794 | ~5k |
| 50k run | ~31,756 | ~7,939 | 7,869 | ~50k |
| 100k run | ~63,513 | ~15,878 | 15,941 | ~100k |

All splits use balanced 1:1 clone/non-clone sampling. Test set is fixed and held out from training.

---

## 3. Model Architecture — GALLa

**GALLa** (Graph ALigned Large Language model) bridges graph-structured code representations with Large Language Models.

### Three Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        GALLa Pipeline                           │
│                                                                 │
│  Java CFG+DFG ──┐                                               │
│                 ├──► Combined Graph                             │
│  Python CFG+DFG─┘         │                                     │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │   DUPLEX GNN (3-layer)  │                        │
│              │   GAT-based             │                        │
│              │   Input:  200-dim       │                        │
│              │   Hidden: 512-dim       │                        │
│              │   Output: 512-dim       │                        │
│              │   Params: ~1.8M         │                        │
│              └───────────┬─────────────┘                        │
│                          │ node embeddings                      │
│                          ▼                                      │
│              ┌─────────────────────────┐                        │
│              │   Cross-Attention       │                        │
│              │   Adapter               │                        │
│              │   64 learnable queries  │                        │
│              │   → 64 graph tokens     │                        │
│              │   Params: ~6.4M         │                        │
│              └───────────┬─────────────┘                        │
│                          │ 64 × 1536-dim tokens                 │
│                          ▼                                      │
│   Prompt: [human] [graph_pad × 64] Are these programs           │
│           clones? [bot]                                         │
│                          │                                      │
│              ┌─────────────────────────┐                        │
│              │  Qwen2.5-Coder-1.5B    │                        │
│              │  + LoRA (rank=16)       │                        │
│              │  Params: 1.56B total    │                        │
│              │  Trained: LoRA only     │                        │
│              └───────────┬─────────────┘                        │
│                          │                                      │
│              logit("yes") vs logit("no")                        │
│                          │                                      │
│                    Prediction                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. DUPLEX GNN
- Architecture: 3-layer Graph Attention Network (GAT)
- Processes the combined Java+Python CFG+DFG
- Input: 200-dim Bytecode2Vec node embeddings
- Output: 512-dim node representations
- Parameters: ~1.8M

#### 2. Cross-Attention Adapter
- 64 learnable query vectors (one per graph token)
- Multi-head cross-attention: queries attend to GNN node outputs
- Projects graph into 64 soft tokens in LM embedding space (1536-dim)
- Parameters: ~6.4M

#### 3. Qwen2.5-Coder-1.5B + LoRA
- Pre-trained code LLM (1.56B parameters)
- LoRA fine-tuning: rank=16, alpha=16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
- Only LoRA weights (~few million params) are updated during training
- Base LM weights remain frozen

### Inference
```
logit("yes") > logit("no")  →  clone
logit("yes") < logit("no")  →  non-clone
```
Next-token prediction — no generation needed.

---

## 4. Training Details

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | Qwen2.5-Coder-1.5B | Strong code understanding |
| LoRA rank | 16 | Balance efficiency vs capacity |
| LoRA alpha | 16 | Standard scaling |
| Learning rate | 5e-6 | Conservative for LoRA |
| Min LR | 1e-7 | Cosine decay floor |
| LR scheduler | Cosine | Smooth decay |
| Warmup steps | 10 | Quick warmup for LoRA |
| Batch size | 1 (per device) | Large graphs, GPU memory constraint |
| Gradient accumulation | 4 | Effective batch size = 4 |
| Weight decay | 0.1 | Regularization |
| Epochs | 5 | With early stopping |
| Early stopping | 5 stalls | Stop if val loss doesn't improve |
| Sequence length | 2048 | Covers prompt + graph tokens |
| Graph tokens | 64 | Injected into LM input |
| Graph hidden dim | 512 | GNN hidden size |

### Training Setup
- **Hardware:** Single NVIDIA Tesla P100-PCIE-12GB GPU (CUDA 12.4)
- **Framework:** PyTorch 2.3 + DGL 2.3 + HuggingFace Transformers + PEFT
- **Acceleration:** Accelerate (single GPU)
- **Checkpoint:** Every 5000 steps; best checkpoint (lowest val loss) saved separately
- **Training time:** ~12 hrs (50k), ~60 hrs (100k warm-started from 50k)

### What is Trained vs Frozen

| Component | Status | Parameters |
|-----------|--------|------------|
| Qwen2.5-Coder-1.5B base | **Frozen** | 1.56B |
| LoRA adapters | **Trained** | ~few M |
| DUPLEX GNN | **Trained** | ~1.8M |
| Cross-attention Adapter | **Trained** | ~6.4M |

### Loss
Cross-entropy loss on next-token prediction at the "yes"/"no" position.

---

## 5. Evaluation Metrics

Binary classification metrics:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP+TN) / Total | Overall correct predictions |
| Precision | TP / (TP+FP) | Of predicted clones, how many are real |
| Recall | TP / (TP+FN) | Of real clones, how many were found |
| F1 Score | 2×P×R / (P+R) | Harmonic mean of Precision & Recall |

**Confusion Matrix:**
```
                 Predicted NO    Predicted YES
Actual NO            TN               FP
Actual YES           FN               TP
```

- **TP** — correctly identified clones
- **TN** — correctly identified non-clones
- **FP** — false alarms (non-clones predicted as clones)
- **FN** — missed clones

---

## 6. Experimental Results

### Experiment 1 — Smoke Test (5k pairs)

| Split | Samples |
|-------|---------|
| Train | ~3,575 |
| Validation | ~397 |
| Test | 794 |

| Metric | Score |
|--------|-------|
| Accuracy | 0.6474 |
| Precision | 0.6240 |
| Recall | 0.6373 |
| **F1 Score** | **0.6306** |

**Confusion Matrix:**
```
                Pred NO   Pred YES
  Actual NO       275       144
  Actual YES      136       239
```

> Limited by small training data — model is learning but needs more examples.

---

### Experiment 2 — Full Run (50k pairs)

| Split | Samples |
|-------|---------|
| Train | ~31,756 |
| Validation | ~7,939 |
| Test | 7,869 |

| Metric | Score |
|--------|-------|
| Accuracy | 0.6860 |
| Precision | 0.6847 |
| Recall | 0.6545 |
| **F1 Score** | **0.6693** |

**Confusion Matrix:**
```
                Pred NO   Pred YES
  Actual NO      2898      1151
  Actual YES     1320      2500
```

> Note: Validation was disabled during this run — final checkpoint used instead of best checkpoint. Experiment 3 (100k) with validation enabled achieved F1 = **0.6917**.

---

### Experiment 3 — Extended Run (100k pairs, warm-started from 50k)

| Split | Samples |
|-------|---------|
| Train | ~63,513 |
| Validation | ~15,878 |
| Test | 15,941 |

| Metric | Score |
|--------|-------|
| Accuracy | 0.7059 |
| Precision | 0.6986 |
| Recall | 0.6849 |
| **F1 Score** | **0.6917** |

**Confusion Matrix:**
```
                Pred NO   Pred YES
  Actual NO       5992      2269
  Actual YES      2420      5260
```

> Warm-started from the 50k final checkpoint. Best checkpoint selected by lowest validation loss (eval_loss = 0.2931). F1 improves from 0.6693 → **0.6917** (+3.3% relative) with doubled training data.

---

## 7. Key Design Decisions

| Decision | Alternative | Why We Chose This |
|----------|------------|-------------------|
| Bytecode2Vec embeddings (pre-computed) | Train embeddings from scratch | Leverages pre-trained opcode semantics |
| Combined Java+Python graph | Separate graphs + fusion | Simpler, GNN naturally learns cross-language structure |
| GALLa (GNN + Adapter + LLM) | Text-only LLM | Graph tokens inject structural info the LLM can't infer from code text |
| LoRA fine-tuning | Full fine-tuning | Memory efficient, prevents catastrophic forgetting |
| Next-token logit comparison | Generate full answer | Faster inference, no decoding needed |
| Balanced 1:1 sampling | Natural 1:5 ratio | More stable training, avoids class imbalance |

---

## 8. Pipeline Summary

```
Raw Data (AtCoder CSV)
        │
        ▼
prepare_bytecode_clone.py
  - Read Java/Python graph JSONs
  - Mean-pool opcode embeddings per basic block
  - Concatenate Java + Python graphs
  - Write JSONL: {embeddings, edge_index, question, bot, label}
        │
        ▼
run.py (Training)
  - Load JSONL, split 80/20 train/val
  - Build DGL graphs per pair
  - Forward: GNN → Adapter → LM
  - Loss: cross-entropy on yes/no token
  - Save best checkpoint (lowest val loss)
        │
        ▼
evaluate_clone_full.py (Evaluation)
  - Load GNN + Adapter + LoRA weights
  - Run full GALLa inference on test set
  - Compare logit("yes") vs logit("no")
  - Compute Accuracy, Precision, Recall, F1
  - Save results to JSON + CSV
```

---

## 9. Comparison with Baseline

| Method | F1 | Notes |
|--------|-----|-------|
| Bytecode2Vec (original paper) | ~0.85 | Same-language, supervised |
| Text-only LLM (no graph) | ~0.50 | Random — no graph tokens injected |
| GALLa 5k (ours) | 0.6306 | Small data, proof of concept |
| GALLa 50k final checkpoint (ours) | 0.6693 | Validation disabled — not best checkpoint |
| **GALLa 100k best checkpoint (ours)** | **0.6917** | Warm-started from 50k; best checkpoint by val loss |

---

## 10. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10 | Programming language |
| PyTorch 2.3 | Deep learning framework |
| DGL 2.3 | Graph neural network library |
| HuggingFace Transformers | LLM loading & tokenization |
| PEFT (LoRA) | Parameter-efficient fine-tuning |
| Accelerate | Training acceleration |
| Qwen2.5-Coder-1.5B | Base language model |
| Bytecode2Vec | Opcode embeddings (pre-trained) |
| GALLa | Graph-aligned LLM framework (ACL 2025) |

---

## 11. Slide Topics (Presentation Outline)

| Slide | Title | Key Content |
|-------|-------|-------------|
| 1 | Title Slide | Project title, name, guide, institution |
| 2 | Problem Statement | What is cross-language code clone detection? Why is it hard? |
| 3 | Motivation & Applications | Plagiarism detection, code search, software maintenance |
| 4 | Dataset | AtCoder Bytecode2Vec — Java ↔ Python pairs, graph structure, class distribution |
| 5 | Bytecode Graph Representation | CFG+DFG, basic blocks, Bytecode2Vec 200-dim node embeddings |
| 6 | Model Architecture — GALLa Overview | Full pipeline diagram: GNN → Adapter → LLM |
| 7 | Component 1: DUPLEX GNN | 3-layer GAT, input/hidden/output dims, parameter count |
| 8 | Component 2: Cross-Attention Adapter | 64 learnable queries, graph → 64 soft tokens in LM space |
| 9 | Component 3: Qwen2.5-Coder + LoRA | Pre-trained LLM, LoRA rank=16, next-token logit comparison |
| 10 | Training Strategy | What is trained vs frozen, hyperparameters, loss function |
| 11 | Experimental Results | Table: 5k → 50k → 100k — Accuracy, Precision, Recall, F1 |
| 12 | Comparison with Baseline | GALLa vs Text-only LLM vs Bytecode2Vec original paper |
| 13 | Conclusion & Future Work | Summary of findings, limitations, potential next steps |

### Suggested Slide 13 Content — Conclusion & Future Work

**Conclusion:**
- GALLa pipeline successfully adapted for cross-language clone detection using bytecode graphs
- F1 improves consistently with more data: 0.63 (5k) → 0.67 (50k) → **0.69 (100k)**
- Graph tokens from GNN/Adapter provide structural signal the text-only LLM cannot infer
- Cross-language clone detection is significantly harder than same-language (gap vs ~0.85 baseline)

**Future Work:**
- Train on the full 256k usable pairs (currently limited to 100k)
- Add cross-language edges between Java and Python nodes in the combined graph
- Try larger base LMs (Qwen2.5-Coder-7B, CodeLlama-13B)
- Experiment with graph pooling strategies beyond cross-attention (e.g., mean/max pooling)
- Evaluate on other language pairs (C++ ↔ Python, Java ↔ C++)
