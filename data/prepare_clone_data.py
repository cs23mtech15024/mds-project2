"""Prepare BigCloneBench clone detection data for GALLa fine-tuning.

Reads BigCloneBench Java functions and pair labels, parses each function's
AST with tree-sitter, and writes JSONL files in GALLa's instruction format:
  {node_ids, edge_index, question, bot}  <- graph QA format used in ft mode

Smart split strategy:
  - Function-disjoint: no function appears in both train+val and test sets
  - Stratified: yes/no labels balanced in every split
  - Train+val → data_sample/clone/clone_graph.jsonl  (graph_dataset.py does 90/10)
  - Test      → data_sample/clone_test.jsonl

Run from the GALLa root directory:
  python data/prepare_clone_data.py

Adjust BCB_DIR to point to your local CodeXGLUE clone-detection dataset.
"""

import json
import os
import random

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BCB_DIR = (
    "/Users/sumandey/Desktop/IITH/Classes/Sem 6/Project/CodeXGLUE"
    "/Code-Code/Clone-detection-BigCloneBench/dataset"
)
DATA_JSONL = f"{BCB_DIR}/data.jsonl"
TRAIN_TXT  = f"{BCB_DIR}/train.txt"

OUT_TRAIN_DIR = "data_sample/clone"
OUT_TEST_FILE = "data_sample/clone_test.jsonl"

# ── Sample sizes ───────────────────────────────────────────────────────────────
N_TRAIN_VAL = 2000  # train+val pairs  (graph_dataset.py splits 90/10 → 1800/200)
N_TEST      = 200   # held-out test pairs
# Total pairs needed: 2200  (1100 yes + 1100 no)
# We over-sample candidates to allow function-disjoint filtering
CANDIDATE_MULTIPLIER = 5   # collect 5× what we need before filtering

MAX_FUNC_CHARS = 500
MAX_AST_NODES  = 300

# ── GALLa node type list ───────────────────────────────────────────────────────
NODE_TYPES = [
    'assignment expression',         # 0
    'basic block',                   # 1
    'binary expression',             # 2
    'break statement',               # 3
    'call expression',               # 4
    'catch clause',                  # 5
    'class expression',              # 6
    'compile unit',                  # 7
    'conditional expression',        # 8
    'continue statement',            # 9
    'export statement',              # 10
    'for statement',                 # 11
    'function expression',           # 12
    'identifier expression',         # 13
    'if statement',                  # 14
    'import expression',             # 15
    'key value parameter',           # 16
    'literal expression',            # 17
    'member access',                 # 18
    'new expression',                # 19
    'new with type expression',      # 20
    'object expression',             # 21
    'object property',               # 22
    'parameter',                     # 23
    'Python delete',                 # 24
    'Python with',                   # 25
    'Python with expression clause', # 26
    'Python yield expression',       # 27
    'range statement',               # 28
    'return statement',              # 29
    'scope',                         # 30
    'spread collection expression',  # 31
    'spread dictionary expression',  # 32
    'super expression',              # 33
    'switch case',                   # 34
    'switch statement',              # 35
    'this expression',               # 36
    'throw statement',               # 37
    'try statement',                 # 38
    'tuple expression',              # 39
    'unary expression',              # 40
    'variable declaration',          # 41
    'while statement',               # 42
]

JAVA_TO_GALLA = {
    'assignment_expression':            0,
    'augmented_assignment_expression':  0,
    'block':                            1,
    'binary_expression':                2,
    'instanceof_expression':            2,
    'break_statement':                  3,
    'method_invocation':                4,
    'explicit_generic_invocation':      4,
    'catch_clause':                     5,
    'class_declaration':                6,
    'interface_declaration':            6,
    'enum_declaration':                 6,
    'annotation_type_declaration':      6,
    'program':                          7,
    'ternary_expression':               8,
    'continue_statement':               9,
    'for_statement':                    11,
    'enhanced_for_statement':           11,
    'method_declaration':               12,
    'constructor_declaration':          12,
    'identifier':                       13,
    'type_identifier':                  13,
    'scoped_type_identifier':           13,
    'generic_type':                     13,
    'array_type':                       13,
    'if_statement':                     14,
    'import_declaration':               15,
    'formal_parameter':                 23,
    'spread_parameter':                 23,
    'receiver_parameter':               23,
    'string_literal':                   17,
    'integer_literal':                  17,
    'decimal_integer_literal':          17,
    'hex_integer_literal':              17,
    'octal_integer_literal':            17,
    'binary_integer_literal':           17,
    'decimal_floating_point_literal':   17,
    'hex_floating_point_literal':       17,
    'character_literal':                17,
    'null_literal':                     17,
    'true':                             17,
    'false':                            17,
    'void_type':                        17,
    'integral_type':                    17,
    'floating_point_type':              17,
    'boolean_type':                     17,
    'field_access':                     18,
    'method_reference':                 18,
    'array_access':                     18,
    'object_creation_expression':       19,
    'array_creation_expression':        20,
    'class_body':                       30,
    'interface_body':                   30,
    'enum_body':                        30,
    'super':                            33,
    'switch_block_statement_group':     34,
    'switch_rule':                      34,
    'switch_statement':                 35,
    'switch_expression':                35,
    'this':                             36,
    'throw_statement':                  37,
    'try_statement':                    38,
    'try_with_resources_statement':     38,
    'unary_expression':                 40,
    'update_expression':                40,
    'local_variable_declaration':       41,
    'field_declaration':                41,
    'variable_declarator':              41,
    'while_statement':                  42,
    'do_statement':                     42,
    'return_statement':                 29,
}

DEFAULT_NODE_ID = 13


def get_parser():
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
    return Parser(Language(tsjava.language()))


def build_ast_graph(code: str, parser, max_nodes: int = MAX_AST_NODES):
    tree = parser.parse(bytes(code, "utf-8"))
    node_ids, edge_index = [], []

    def visit(node, parent_idx):
        if len(node_ids) >= max_nodes:
            return
        if node.is_named:
            idx = len(node_ids)
            node_ids.append(JAVA_TO_GALLA.get(node.type, DEFAULT_NODE_ID))
            if parent_idx is not None:
                edge_index.append([parent_idx, idx])
            next_parent = idx
        else:
            next_parent = parent_idx
        for child in node.children:
            visit(child, next_parent)

    visit(tree.root_node, None)

    if len(node_ids) < 2:
        node_ids = [DEFAULT_NODE_ID, DEFAULT_NODE_ID]
    if len(edge_index) == 0:
        edge_index = [[0, 1]]
    return node_ids, edge_index


def make_question(func1: str, func2: str) -> str:
    return (
        "# Determine if the following two Java functions are code clones "
        "(semantically equivalent implementations of the same logic)\n\n"
        f"# Function 1\n```java\n{func1.strip()}\n```\n\n"
        f"# Function 2\n```java\n{func2.strip()}\n```\n\n"
        "# Are these two Java functions code clones? Answer yes or no."
    )


def load_function_index(path: str) -> dict:
    index = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            index[str(d["idx"])] = d["func"]
    return index


def collect_candidates(train_txt: str, func_index: dict, n_pos: int, n_neg: int):
    """Collect candidate pairs (idx1, idx2, func1, func2, label)."""
    pos, neg = [], []
    needed = max(n_pos, n_neg)
    with open(train_txt) as f:
        for line in f:
            if len(pos) >= needed and len(neg) >= needed:
                break
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            idx1, idx2, label = parts[0], parts[1], int(parts[2])
            if idx1 not in func_index or idx2 not in func_index:
                continue
            f1 = func_index[idx1][:MAX_FUNC_CHARS]
            f2 = func_index[idx2][:MAX_FUNC_CHARS]
            if len(f1) < 20 or len(f2) < 20:
                continue
            entry = (idx1, idx2, f1, f2, label)
            if label == 1 and len(pos) < needed:
                pos.append(entry)
            elif label == 0 and len(neg) < needed:
                neg.append(entry)
    return pos, neg


def function_disjoint_split(pos_candidates, neg_candidates, n_trainval, n_test):
    """
    Split pairs so no function ID appears in both train+val and test.

    Strategy:
      1. Collect all unique function IDs from candidates.
      2. Randomly assign 80% to train_funcs, 20% to test_funcs.
      3. A pair belongs to train+val only if BOTH its functions are in train_funcs.
      4. A pair belongs to test      only if BOTH its functions are in test_funcs.
      5. Balance yes/no within each split.
    """
    all_candidates = pos_candidates + neg_candidates

    # Collect all unique function IDs
    all_func_ids = set()
    for idx1, idx2, *_ in all_candidates:
        all_func_ids.add(idx1)
        all_func_ids.add(idx2)

    func_ids = list(all_func_ids)
    random.shuffle(func_ids)
    split_point = int(len(func_ids) * 0.8)
    train_funcs = set(func_ids[:split_point])
    test_funcs  = set(func_ids[split_point:])

    trainval_pos, trainval_neg = [], []
    test_pos,     test_neg     = [], []

    for idx1, idx2, f1, f2, label in pos_candidates:
        if idx1 in train_funcs and idx2 in train_funcs:
            trainval_pos.append((idx1, idx2, f1, f2, label))
        elif idx1 in test_funcs and idx2 in test_funcs:
            test_pos.append((idx1, idx2, f1, f2, label))

    for idx1, idx2, f1, f2, label in neg_candidates:
        if idx1 in train_funcs and idx2 in train_funcs:
            trainval_neg.append((idx1, idx2, f1, f2, label))
        elif idx1 in test_funcs and idx2 in test_funcs:
            test_neg.append((idx1, idx2, f1, f2, label))

    # Balanced sampling
    half_trainval = n_trainval // 2
    half_test     = n_test     // 2

    random.shuffle(trainval_pos); random.shuffle(trainval_neg)
    random.shuffle(test_pos);     random.shuffle(test_neg)

    # Keep (idx1, idx2, f1, f2, label) for leakage check, strip to (f1, f2, label) after
    trainval_raw = trainval_pos[:half_trainval] + trainval_neg[:half_trainval]
    test_raw     = test_pos[:half_test]         + test_neg[:half_test]

    random.shuffle(trainval_raw)
    random.shuffle(test_raw)

    # Leakage check using actual function IDs
    train_ids = set()
    for idx1, idx2, *_ in trainval_raw:
        train_ids.add(idx1); train_ids.add(idx2)
    leakage = sum(1 for idx1, idx2, *_ in test_raw
                  if idx1 in train_ids or idx2 in train_ids)

    trainval = [(f1, f2, label) for _, _, f1, f2, label in trainval_raw]
    test     = [(f1, f2, label) for _, _, f1, f2, label in test_raw]

    return trainval, test, {
        "trainval_pos_avail": len(trainval_pos),
        "trainval_neg_avail": len(trainval_neg),
        "test_pos_avail":     len(test_pos),
        "test_neg_avail":     len(test_neg),
        "train_funcs":        len(train_funcs),
        "test_funcs":         len(test_funcs),
        "leakage":            leakage,
    }


def write_jsonl(pairs, path: str, parser, include_label: bool = False):
    written = skipped = 0
    with open(path, "w") as fout:
        for func1, func2, label in pairs:
            try:
                node_ids, edge_index = build_ast_graph(func1, parser)
            except Exception:
                skipped += 1
                continue
            sample = {
                "node_ids":   node_ids,
                "edge_index": edge_index,
                "question":   make_question(func1, func2),
                "bot":        "yes" if label == 1 else "no",
            }
            if include_label:
                sample["label"] = label
            fout.write(json.dumps(sample) + "\n")
            written += 1
    return written, skipped


def main():
    total_needed  = (N_TRAIN_VAL + N_TEST) // 2  # per class
    candidate_per_class = total_needed * CANDIDATE_MULTIPLIER

    print(f"Target: {N_TRAIN_VAL} train+val, {N_TEST} test  ({N_TRAIN_VAL//2} yes + {N_TRAIN_VAL//2} no for train+val)")
    print(f"Collecting {candidate_per_class} candidate pairs per class for disjoint filtering...")

    print("Loading function index...")
    func_index = load_function_index(DATA_JSONL)
    print(f"  {len(func_index)} functions loaded")

    print("Sampling candidate pairs...")
    pos_cands, neg_cands = collect_candidates(
        TRAIN_TXT, func_index,
        n_pos=candidate_per_class,
        n_neg=candidate_per_class,
    )
    print(f"  collected {len(pos_cands)} positive, {len(neg_cands)} negative candidates")

    print("Applying function-disjoint split...")
    trainval_pairs, test_pairs, stats = function_disjoint_split(
        pos_cands, neg_cands, N_TRAIN_VAL, N_TEST
    )

    print(f"  Function pool  : {stats['train_funcs']} train-funcs, {stats['test_funcs']} test-funcs")
    print(f"  Available pairs: trainval pos={stats['trainval_pos_avail']}, neg={stats['trainval_neg_avail']}")
    print(f"                   test    pos={stats['test_pos_avail']},     neg={stats['test_neg_avail']}")
    print(f"  Final          : {len(trainval_pairs)} train+val, {len(test_pairs)} test")
    print(f"  Function leakage check: {stats['leakage']} overlapping pairs (should be 0)")

    print("Initialising tree-sitter Java parser...")
    parser = get_parser()

    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TEST_FILE) or ".", exist_ok=True)

    train_path = os.path.join(OUT_TRAIN_DIR, "clone_graph.jsonl")
    print(f"Writing train+val → {train_path}")
    w, s = write_jsonl(trainval_pairs, train_path, parser, include_label=False)
    print(f"  wrote {w} samples, skipped {s}")

    print(f"Writing test → {OUT_TEST_FILE}")
    w, s = write_jsonl(test_pairs, OUT_TEST_FILE, parser, include_label=True)
    print(f"  wrote {w} samples, skipped {s}")

    # Print final label distribution
    for path, name in [(train_path, "Train+Val"), (OUT_TEST_FILE, "Test")]:
        yes = no = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["bot"] == "yes": yes += 1
                else: no += 1
        print(f"  {name}: yes={yes}, no={no}, total={yes+no}")

    print("Done.")


if __name__ == "__main__":
    main()
