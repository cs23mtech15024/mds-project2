"""Unified clone-detection data preparation for GALLa.

Handles three input scenarios via --mode flag:

  1. source      — raw Java source code → tree-sitter AST → node_ids + edge_index
  2. bytecode    — compiled .class files → javap disassembly → CFG → node_ids + edge_index
  3. precomputed — pre-built graph (AST/CFG/DFG) → direct mapping → node_ids + edge_index

All three produce the same output JSONL format consumed by GALLa training/eval.

────────────────────────────────────────────────────────────────────────────────
Input JSONL format per mode
────────────────────────────────────────────────────────────────────────────────

mode=source:
  {"func1": "<java source>", "func2": "<java source>", "label": 1}

mode=bytecode:
  {"class1": "path/to/A.class", "class2": "path/to/B.class", "label": 0}

mode=precomputed:
  {
    "nodes1": [{"type": "if_statement"}, {"type": "return"}, ...],
    "edges1": [[0,1], [1,2], ...],
    "nodes2": [...],   # optional — only used as text context
    "edges2": [...],   # optional
    "text1":  "<method text or description>",
    "text2":  "<method text or description>",
    "label":  1
  }
  node type strings are mapped via CUSTOM_TO_GALLA (edit as needed).
  Alternatively pass node_ids directly:
  {"node_ids1": [7,12,13,...], "edge_index1": [[0,1],...], "text1": "...", ...}

────────────────────────────────────────────────────────────────────────────────
Usage (from GALLa root)
────────────────────────────────────────────────────────────────────────────────
  python data/prepare_clone_unified.py \
      --mode      source \
      --input     data/my_pairs.jsonl \
      --out_train data_sample/clone/clone_graph.jsonl \
      --out_test  data_sample/clone_test.jsonl

  python data/prepare_clone_unified.py \
      --mode      bytecode \
      --input     data/bytecode_pairs.jsonl \
      --out_train data_sample/clone/clone_graph.jsonl \
      --out_test  data_sample/clone_test.jsonl

  python data/prepare_clone_unified.py \
      --mode      precomputed \
      --input     data/precomputed_graphs.jsonl \
      --out_train data_sample/clone/clone_graph.jsonl \
      --out_test  data_sample/clone_test.jsonl
"""

import argparse
import json
import os
import random
import re
import subprocess

random.seed(42)

MAX_NODES = 300

# ── GALLa node type ID → name mapping (for reference) ────────────────────────
# Full list in arguments.py. Key IDs used below:
#   2=binary_expr  4=call  7=compile_unit  12=method  13=identifier
#   14=if  17=literal  18=member_access  19=new  20=array_new
#   23=parameter  29=return  35=switch  37=throw  38=try
#   41=variable  42=while

DEFAULT_NODE_ID = 13   # identifier — fallback for anything unmapped


# ════════════════════════════════════════════════════════════════════════════════
# MODE 1 — SOURCE CODE  (tree-sitter Java AST)
# ════════════════════════════════════════════════════════════════════════════════

JAVA_TO_GALLA = {
    'assignment_expression': 0, 'augmented_assignment_expression': 0,
    'block': 1,
    'binary_expression': 2, 'instanceof_expression': 2,
    'break_statement': 3,
    'method_invocation': 4, 'explicit_generic_invocation': 4,
    'catch_clause': 5,
    'class_declaration': 6, 'interface_declaration': 6,
    'enum_declaration': 6, 'annotation_type_declaration': 6,
    'program': 7,
    'ternary_expression': 8,
    'continue_statement': 9,
    'for_statement': 11, 'enhanced_for_statement': 11,
    'method_declaration': 12, 'constructor_declaration': 12,
    'identifier': 13, 'type_identifier': 13, 'scoped_type_identifier': 13,
    'generic_type': 13, 'array_type': 13,
    'if_statement': 14,
    'import_declaration': 15,
    'formal_parameter': 23, 'spread_parameter': 23, 'receiver_parameter': 23,
    'string_literal': 17, 'integer_literal': 17, 'decimal_integer_literal': 17,
    'hex_integer_literal': 17, 'decimal_floating_point_literal': 17,
    'character_literal': 17, 'null_literal': 17,
    'true': 17, 'false': 17, 'void_type': 17,
    'integral_type': 17, 'floating_point_type': 17, 'boolean_type': 17,
    'field_access': 18, 'method_reference': 18, 'array_access': 18,
    'object_creation_expression': 19,
    'array_creation_expression': 20,
    'class_body': 30, 'interface_body': 30, 'enum_body': 30,
    'super': 33,
    'switch_block_statement_group': 34, 'switch_rule': 34,
    'switch_statement': 35, 'switch_expression': 35,
    'this': 36,
    'throw_statement': 37,
    'try_statement': 38, 'try_with_resources_statement': 38,
    'unary_expression': 40, 'update_expression': 40,
    'local_variable_declaration': 41, 'field_declaration': 41,
    'variable_declarator': 41,
    'while_statement': 42, 'do_statement': 42,
    'return_statement': 29,
}


def _get_java_parser():
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
    return Parser(Language(tsjava.language()))


def build_source_graph(code: str, parser, max_nodes=MAX_NODES):
    """Java source code → (node_ids, edge_index) via tree-sitter AST."""
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
    return _ensure_valid(node_ids, edge_index)


# ════════════════════════════════════════════════════════════════════════════════
# MODE 2 — BYTECODE  (javap disassembly → instruction graph)
# ════════════════════════════════════════════════════════════════════════════════

OPCODE_TO_GALLA = {
    # Calls
    "invokevirtual": 4, "invokestatic": 4, "invokeinterface": 4,
    "invokespecial": 4, "invokedynamic": 4,
    # Arithmetic / compare
    "iadd": 2, "isub": 2, "imul": 2, "idiv": 2, "irem": 2,
    "ladd": 2, "lsub": 2, "lmul": 2, "ldiv": 2, "lrem": 2,
    "fadd": 2, "fsub": 2, "fmul": 2, "fdiv": 2, "frem": 2,
    "dadd": 2, "dsub": 2, "dmul": 2, "ddiv": 2, "drem": 2,
    "iand": 2, "ior": 2, "ixor": 2, "ishl": 2, "ishr": 2, "iushr": 2,
    "land": 2, "lor": 2, "lxor": 2,
    "lcmp": 2, "fcmpl": 2, "fcmpg": 2, "dcmpl": 2, "dcmpg": 2,
    "instanceof": 2, "checkcast": 2,
    # Conditional branches
    "ifeq": 14, "ifne": 14, "iflt": 14, "ifge": 14, "ifgt": 14, "ifle": 14,
    "if_icmpeq": 14, "if_icmpne": 14, "if_icmplt": 14,
    "if_icmpge": 14, "if_icmpgt": 14, "if_icmple": 14,
    "if_acmpeq": 14, "if_acmpne": 14, "ifnull": 14, "ifnonnull": 14,
    # Loops / unconditional jump
    "goto": 42, "goto_w": 42, "jsr": 42, "jsr_w": 42, "ret": 42,
    # Switch
    "tableswitch": 35, "lookupswitch": 35,
    # Return
    "return": 29, "ireturn": 29, "areturn": 29,
    "lreturn": 29, "freturn": 29, "dreturn": 29,
    # Throw
    "athrow": 37,
    # Load
    "iload": 13, "iload_0": 13, "iload_1": 13, "iload_2": 13, "iload_3": 13,
    "aload": 13, "aload_0": 13, "aload_1": 13, "aload_2": 13, "aload_3": 13,
    "lload": 13, "fload": 13, "dload": 13,
    "aaload": 13, "iaload": 13, "arraylength": 13,
    # Store
    "istore": 41, "istore_0": 41, "istore_1": 41, "istore_2": 41, "istore_3": 41,
    "astore": 41, "astore_0": 41, "astore_1": 41, "astore_2": 41, "astore_3": 41,
    "lstore": 41, "fstore": 41, "dstore": 41, "aastore": 41, "iastore": 41,
    # Constants
    "ldc": 17, "ldc_w": 17, "ldc2_w": 17, "bipush": 17, "sipush": 17,
    "iconst_m1": 17, "iconst_0": 17, "iconst_1": 17, "iconst_2": 17,
    "iconst_3": 17, "iconst_4": 17, "iconst_5": 17,
    "lconst_0": 17, "lconst_1": 17, "fconst_0": 17, "fconst_1": 17,
    "dconst_0": 17, "dconst_1": 17, "aconst_null": 17,
    # Object / array creation
    "new": 19, "newarray": 20, "anewarray": 20, "multianewarray": 20,
    # Field access
    "getfield": 18, "putfield": 18, "getstatic": 18, "putstatic": 18,
}

_INSTR_RE = re.compile(r"^\s+(\d+):\s+(\w+)", re.MULTILINE)


def _disassemble(class_path: str) -> str:
    result = subprocess.run(
        ["javap", "-c", "-p", class_path],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:200])
    return result.stdout


def build_bytecode_graph(class_path: str, max_nodes=MAX_NODES):
    """Java .class file → (node_ids, edge_index, disassembly_text)."""
    javap_out = _disassemble(class_path)
    instrs    = [(int(m.group(1)), m.group(2).lower())
                 for m in _INSTR_RE.finditer(javap_out)]
    instrs    = instrs[:max_nodes]

    node_ids = [OPCODE_TO_GALLA.get(op, DEFAULT_NODE_ID) for _, op in instrs]
    edge_index = [[i, i + 1] for i in range(len(node_ids) - 1)]

    node_ids, edge_index = _ensure_valid(node_ids, edge_index)
    return node_ids, edge_index, javap_out


# ════════════════════════════════════════════════════════════════════════════════
# MODE 3 — PRE-COMPUTED GRAPH  (professor-supplied AST / CFG / DFG)
# ════════════════════════════════════════════════════════════════════════════════

# Edit this mapping to match whatever node type names the professor uses.
# Keys = professor's node type strings, Values = GALLa node type IDs (0-42).
CUSTOM_TO_GALLA = {
    # AST-style names
    "method":            12,
    "function":          12,
    "if":                14,
    "if_statement":      14,
    "while":             42,
    "while_statement":   42,
    "for":               11,
    "for_statement":     11,
    "return":            29,
    "return_statement":  29,
    "call":               4,
    "call_expression":    4,
    "invoke":             4,
    "variable":          41,
    "assignment":         0,
    "binary":             2,
    "literal":           17,
    "identifier":        13,
    "parameter":         23,
    "block":              1,
    "try":               38,
    "catch":              5,
    "throw":             37,
    "switch":            35,
    "new":               19,
    "field":             18,
    # CFG-style names
    "basic_block":        1,
    "branch":            14,
    "entry":             12,
    "exit":              29,
    "loop":              42,
}


def build_precomputed_graph(record: dict):
    """
    Read graph directly from a pre-computed record.

    Accepts two sub-formats:
      A) node_ids already as integers:
         {"node_ids1": [7,12,...], "edge_index1": [[0,1],...]}

      B) node type strings to be mapped:
         {"nodes1": [{"type": "if_statement"}, ...], "edges1": [[0,1],...]}
    """
    # Sub-format A — IDs already provided
    if "node_ids1" in record:
        node_ids   = record["node_ids1"][:MAX_NODES]
        edge_index = [e for e in record.get("edge_index1", [])
                      if e[0] < len(node_ids) and e[1] < len(node_ids)]
        return _ensure_valid(node_ids, edge_index)

    # Sub-format B — type strings to be mapped
    nodes = record.get("nodes1", [])[:MAX_NODES]
    node_ids = [
        CUSTOM_TO_GALLA.get(
            n.get("type", "").lower().replace(" ", "_"), DEFAULT_NODE_ID
        )
        for n in nodes
    ]
    edge_index = [e for e in record.get("edges1", [])
                  if e[0] < len(node_ids) and e[1] < len(node_ids)]
    return _ensure_valid(node_ids, edge_index)


# ════════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════════

def _ensure_valid(node_ids, edge_index):
    if len(node_ids) < 2:
        node_ids  = list(node_ids) + [DEFAULT_NODE_ID] * (2 - len(node_ids))
    if not edge_index:
        edge_index = [[0, 1]]
    return node_ids, edge_index


def make_question(text1: str, text2: str, mode: str) -> str:
    label = "bytecode methods" if mode == "bytecode" else "Java functions"
    t1 = text1.strip()[:800]
    t2 = text2.strip()[:800]
    return (
        f"# Determine if the following two {label} are code clones "
        "(semantically equivalent implementations of the same logic)\n\n"
        f"# Method 1\n```\n{t1}\n```\n\n"
        f"# Method 2\n```\n{t2}\n```\n\n"
        "# Are these two methods code clones? Answer yes or no."
    )


def write_jsonl(samples, path: str, include_label: bool):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    with open(path, "w") as f:
        for node_ids, edge_index, text1, text2, label, mode in samples:
            sample = {
                "node_ids":   node_ids,
                "edge_index": edge_index,
                "question":   make_question(text1, text2, mode),
                "bot":        "yes" if label == 1 else "no",
            }
            if include_label:
                sample["label"] = label
            f.write(json.dumps(sample) + "\n")
            written += 1
    return written


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       required=True,
                        choices=["source", "bytecode", "precomputed"],
                        help="Input type")
    parser.add_argument("--input",      required=True,
                        help="JSONL file with pairs + labels")
    parser.add_argument("--out_train",  default="data_sample/clone/clone_graph.jsonl")
    parser.add_argument("--out_test",   default="data_sample/clone_test.jsonl")
    parser.add_argument("--test_split", type=float, default=0.1)
    args = parser.parse_args()

    # Load input
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} pairs  (mode={args.mode})")

    # Initialize parser for source mode
    java_parser = None
    if args.mode == "source":
        print("Initialising tree-sitter Java parser...")
        java_parser = _get_java_parser()

    # Process each pair
    processed = []
    for i, rec in enumerate(records):
        try:
            if args.mode == "source":
                node_ids, edge_index = build_source_graph(rec["func1"], java_parser)
                text1, text2 = rec["func1"], rec["func2"]

            elif args.mode == "bytecode":
                node_ids, edge_index, text1 = build_bytecode_graph(rec["class1"])
                _, _, text2                 = build_bytecode_graph(rec["class2"])

            else:  # precomputed
                node_ids, edge_index = build_precomputed_graph(rec)
                text1 = rec.get("text1", "Method 1")
                text2 = rec.get("text2", "Method 2")

            processed.append((node_ids, edge_index, text1, text2,
                               int(rec["label"]), args.mode))

        except Exception as e:
            print(f"  [SKIP] pair {i}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(records)}")

    print(f"Successfully processed {len(processed)} pairs")

    # Stratified train / test split
    random.shuffle(processed)
    n_test   = max(1, int(len(processed) * args.test_split))
    test     = processed[:n_test]
    trainval = processed[n_test:]

    w = write_jsonl(trainval, args.out_train, include_label=False)
    print(f"Train+Val : {w} samples → {args.out_train}")

    w = write_jsonl(test, args.out_test, include_label=True)
    print(f"Test      : {w} samples → {args.out_test}")

    for path, name in [(args.out_train, "Train+Val"), (args.out_test, "Test")]:
        yes = no = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["bot"] == "yes": yes += 1
                else: no += 1
        print(f"  {name}: yes={yes}  no={no}  total={yes+no}")


if __name__ == "__main__":
    main()
