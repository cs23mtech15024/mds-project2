# GALLa Architecture — Stage by Stage

---

## Stage 1 — Graph Pretrain

### Purpose
Train the GNN and Adapter to convert code graph representations (IR) into embeddings that the LLM can understand. The LLM stays frozen — it acts only as a target/teacher.

### Framework Components

| Component | Role | Trained? |
|-----------|------|----------|
| GNN (DUPLEX) | Reads graph IR (node_ids + edge_index), learns structural patterns | YES |
| Adapter (Cross-Attention) | Maps GNN output → LLM token space | YES |
| LLM (Qwen2.5) | Reconstructs source code from graph tokens | NO (frozen) |

### Input
```
node_ids    → type of each node (e.g., call expression, for statement)
edge_index  → connections between nodes (graph structure)
source      → original source code (used as reconstruction target)
```

### What the GNN Learns
- **AST input** → hierarchical structure of code (how code is organized syntactically)
- **DFG input** → data flow patterns (how variables and data move through the program)

### Training Task
```
Graph (AST/DFG) ──GNN──► Graph Embeddings ──Adapter──► Graph Tokens ──LLM──► Reconstructed Source Code
                                                                                        ↓
                                                                              Loss vs original source
                                                                                        ↓
                                                                          Backprop → updates GNN + Adapter only
```

### Output
- Trained GNN + Adapter weights (saved as checkpoint)
- These weights encode deep understanding of code structure

### Use Case: IR to Embeddings
After Stage 1, the model can convert any code graph into a rich embedding:
```
Code → Parse → AST/DFG → GNN → Adapter → Embedding vector
```
This embedding captures both syntax and data flow — ideal for code search and retrieval (IR tasks).

---

## Stage 2 — Instruction Finetuning

### Purpose
Make the model useful for real-world coding tasks by training on graph QA and downstream code tasks simultaneously. The LLM is now trainable via LoRA.

### Framework Components

| Component | Role | Trained? |
|-----------|------|----------|
| GNN (DUPLEX) | Same as Stage 1, initialized from Stage 1 checkpoint | YES |
| Adapter (Cross-Attention) | Same as Stage 1, initialized from Stage 1 checkpoint | YES |
| LLM (Qwen2.5) | Generates answers to questions and code tasks | YES (via LoRA) |

### Input — Two Data Types

**Graph QA data** (`instruction_graph.jsonl`):
```
node_ids    → graph structure
edge_index  → graph connections
question    → "What is the type of this node?"
bot         → "Call expression."
```

**Downstream task data** (`instruction_downstream.jsonl`):
```
human  → "# Fix the bug in the following Java program\n# Buggy\n<code>\n# Fixed\n"
bot    → "<fixed code>"
```

### Training Task
```
Graph QA data ──GNN+Adapter──► Graph Tokens ──┐
                                               ├──► LLM (LoRA) ──► Answer
Downstream data ──Tokenizer──► Text Tokens ───┘         ↓
                                                   Loss (two types)
                                                   training_loss    → graph QA
                                                   training_loss_ft → downstream tasks
                                                         ↓
                                              Backprop → updates GNN + Adapter + LoRA
```

### What the Model Learns
- **From graph QA**: answer questions about code structure (node types, parent/child relationships, edge prediction)
- **From downstream tasks**: perform real software engineering tasks (bug fixing, code translation)
- **Combined effect**: graph understanding improves downstream task performance

### Output
- Fine-tuned LLM saved as a standard HuggingFace model
- No GNN or Adapter needed at inference — graph knowledge is baked into LLM weights

---

## Stage 3 — Inference

### Purpose
Use the fine-tuned model for real-world tasks. At this stage only the LLM is needed — no GNN, no Adapter.

### Why No GNN at Inference?
The graph structural knowledge was transferred into the LLM weights during Stage 1 and Stage 2. The LLM learned to reason about code structure implicitly, so it no longer needs the graph as explicit input.

### Prompt Format
Must match the format used in `instruction_downstream.jsonl` during training:
```
<s>human
# Fix the bug in the following Java program

# Buggy
<buggy code>

# Fixed
<s>bot
<model generates fixed code here>
```

### Loading the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "output/instruction/qwen2.5-coder-1.5b/final_step_X",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
```

### Inference for Bug Fix
```python
prompt = (
    "<s>human\n"
    "# Fix the bug in the following Java program\n\n"
    "# Buggy\n<your buggy code>\n\n"
    "# Fixed\n"
    "<s>bot\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Inference for Embeddings (IR use case)
```python
inputs = tokenizer("your java code", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1)  # mean pooling
```

---

## Summary

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Goal | Learn code structure from graphs | Apply graph knowledge to real tasks | Use the model |
| GNN | Trained | Trained | Not needed |
| Adapter | Trained | Trained | Not needed |
| LLM | Frozen | Trained (LoRA) | Standard HuggingFace |
| Input | node_ids, edge_index, source | Graph QA + Bug-fix pairs | Buggy code prompt |
| Output | GNN + Adapter checkpoint | Fine-tuned LLM | Fixed code / Embedding |
| Use for | Code embeddings / IR | Code generation tasks | Production inference |
