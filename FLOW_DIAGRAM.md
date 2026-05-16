# GALLa Training Flow

---

## STAGE 1 — Graph Pretrain

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                               │
│         Java/Python Code + Graph (AST / DFG)                   │
│         Fields: node_ids, edge_index, source                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GNN ENCODER (DUPLEX)                        │
│         Reads node_ids + edge_index                             │
│         Learns structural patterns from the graph               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼  Graph Embeddings
┌─────────────────────────────────────────────────────────────────┐
│                  ADAPTER (Cross-Attention)                      │
│         Maps graph embeddings → LLM token space                 │
│         Produces graph tokens the LLM can understand            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼  Graph Tokens
┌─────────────────────────────────────────────────────────────────┐
│                   LLM  [*** FROZEN ***]                         │
│                   (Qwen2.5-Coder-1.5B)                         │
│         Task: Reconstruct source code from graph tokens         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                             │
│         Compare predicted code vs actual source code            │
│         Backprop updates GNN + Adapter only (LLM stays frozen)  │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: Trained GNN + Adapter checkpoint
        saved to → output/qwen2.5-coder-1.5b/final_step_X
```

---

## STAGE 2 — Instruction Finetuning

```
┌──────────────────────────────┐   ┌──────────────────────────────┐
│   instruction_graph.jsonl    │   │ instruction_downstream.jsonl  │
│                              │   │                              │
│  node_ids, edge_index,       │   │  human: "Fix this buggy      │
│  question, bot               │   │          Java program..."    │
│                              │   │  bot:   "<fixed code>"       │
│  Graph QA tasks              │   │                              │
│  (node type, parent,         │   │  Real code tasks             │
│   children, edge prediction) │   │  (Java bug-fix)              │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────┐       ┌──────────────────────────────┐
│  GNN + Adapter           │       │  Text Encoder                │
│  (loaded from Stage 1    │       │  (tokenizer)                 │
│   checkpoint)            │       │                              │
└──────────────┬───────────┘       └──────────────┬───────────────┘
               │                                   │
               └──────────────┬────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM with LoRA  [*** TRAINABLE ***]             │
│                  (Qwen2.5-Coder-1.5B)                          │
│                                                                 │
│         Trained on both graph QA + downstream tasks together    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                             │
│    training_loss    → graph QA answers                          │
│    training_loss_ft → downstream task answers                   │
│    Backprop updates GNN + Adapter + LLM (via LoRA)             │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: Fine-tuned model checkpoint
        saved to → output/instruction/qwen2.5-coder-1.5b/final_step_X
```

---

## INFERENCE

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│   "# Fix the bug in the following Java program                  │
│    # Buggy                                                      │
│    <buggy code>                                                 │
│    # Fixed"                                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Fine-tuned LLM (standard HuggingFace model)        │
│         No GNN or Adapter needed at inference time              │
│         Graph knowledge is baked into LLM weights               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│                    <fixed Java code>                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

| | Stage 1 | Stage 2 |
|---|---|---|
| What trains | GNN + Adapter | GNN + Adapter + LLM (LoRA) |
| LLM | Frozen | Trainable |
| Data | Graph only (node_ids, edge_index, source) | Graph QA + Downstream tasks |
| Goal | Learn code structure from graphs | Apply graph understanding to real tasks |
| Output | Trained GNN + Adapter | Full fine-tuned model |
