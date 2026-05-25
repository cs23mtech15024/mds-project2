"""
Prepare Bytecode2Vec AtCoder clone-detection data for the GALLa pipeline.

Reads:
  - JavaJson / PyJson per-file graphs (CFG + DFG, pre-computed 200-dim embeddings)
  - shuffled_final_{train,test}_bytecode2vec.csv (index1, index2, label)

Writes two JSONL files (train / test) in the GALLa graph-ft format:
  {embeddings, edge_index, question, bot}

Graph per sample = Java graph + Python graph concatenated (Python node IDs offset
by Java node count). All methods in a file are merged; intra-method CFG and DFG
edges are included; no inter-method call edges for now.

Usage:
  python data/prepare_bytecode_clone.py \
      --java_dir  /Pramana/.../JavaJson \
      --py_dir    /Pramana/.../PyJson \
      --train_csv /Pramana/.../shuffled_final_train_bytecode2vec.csv \
      --test_csv  /Pramana/.../shuffled_final_test_bytecode2vec.csv \
      --out_dir   data_sample/bytecode_clone \
      --n_train   1000 \
      --n_test    200
"""

import os, json, csv, argparse, random
import numpy as np

random.seed(42)

JAVA_DIR = "/Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/Atcoder/atcoder/src/javaPyFlowInfoJson/JavaJson"
PY_DIR   = "/Pramana/ByteCode2Vec/Code_Clone_Detection_Dataset/Dataset/Atcoder/atcoder/src/javaPyFlowInfoJson/PyJson"
TRAIN_CSV = "/Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/shuffled_final_train_bytecode2vec.csv"
TEST_CSV  = "/Pramana/ByteCode2Vec/Baselines/Bytecode2Vec/Atcoder_jp/shuffled_final_test_bytecode2vec.csv"


# ── index → filename maps ──────────────────────────────────────────────────────

def build_index_maps(java_dir, py_dir):
    java_map, py_map = {}, {}
    for f in os.listdir(java_dir):
        if not f.endswith('.json'):
            continue
        parts = f.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            java_map[parts[1]] = os.path.join(java_dir, f)
    for f in os.listdir(py_dir):
        if not f.endswith('.json'):
            continue
        parts = f.replace('.json', '').split('_')
        for p in reversed(parts):
            if p.isdigit() and len(p) >= 5:
                py_map[p] = os.path.join(py_dir, f)
                break
    return java_map, py_map


# ── per-file graph extraction ─────────────────────────────────────────────────

def mean_pool(embedding_list):
    """Mean-pool a list of 200-dim opcode embeddings → one 200-dim block embedding."""
    arr = np.array(embedding_list, dtype=np.float32)
    return arr.mean(axis=0).tolist()


def extract_file_graph(json_path, use_dfg=True):
    """
    Load a Java or Python JSON file and return:
      embeddings : list of 200-dim vectors, one per basic block (all methods merged)
      edge_index : list of [src, dst] pairs
    Returns (None, None) on failure or empty graph.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None, None

    all_embeddings = []
    all_edges = []
    block_offset = 0

    for method_entry in data:
        key = list(method_entry.keys())[0]
        graphs = method_entry[key]
        if not graphs:
            continue
        g = graphs[0]
        nodes = g.get('Nodes', [])
        if not nodes:
            continue

        # node embeddings: mean-pool opcodes per block
        block_embs = []
        for node in nodes:
            embs = node.get('embeddings', [])
            if not embs:
                # fallback: zero vector
                block_embs.append([0.0] * 200)
            else:
                block_embs.append(mean_pool(embs))

        n_blocks = len(block_embs)
        all_embeddings.extend(block_embs)

        # CFG edges
        cfg = g.get('Edges_CFG', {})
        for src_str, dsts in cfg.items():
            src = int(src_str) + block_offset
            for dst in dsts:
                if isinstance(dst, int) and dst < n_blocks:
                    all_edges.append([src, dst + block_offset])

        # DFG edges (target is block id, strip variable name)
        if use_dfg:
            dfg = g.get('Edges_DFG', {})
            for src_str, edges in dfg.items():
                src = int(src_str) + block_offset
                for edge in edges:
                    if isinstance(edge, list) and len(edge) == 2:
                        dst = edge[1]
                        if isinstance(dst, int) and dst < n_blocks:
                            all_edges.append([src, dst + block_offset])

        block_offset += n_blocks

    if len(all_embeddings) < 2:
        return None, None

    # deduplicate edges, add self-loop on node 0 if no edges
    edge_set = list({(e[0], e[1]) for e in all_edges})
    if not edge_set:
        edge_set = [(0, 0)]
    edge_index = [[u, v] for u, v in edge_set]

    return all_embeddings, edge_index


# ── pair builder ──────────────────────────────────────────────────────────────

def build_pair_graph(java_path, py_path):
    """
    Merge Java and Python file graphs into one graph.
    Python node IDs are offset by the number of Java blocks.
    Returns (embeddings, edge_index) or (None, None).
    """
    j_embs, j_edges = extract_file_graph(java_path)
    p_embs, p_edges = extract_file_graph(py_path)

    if j_embs is None or p_embs is None:
        return None, None

    offset = len(j_embs)
    combined_embs = j_embs + p_embs
    combined_edges = j_edges + [[u + offset, v + offset] for u, v in p_edges]

    return combined_embs, combined_edges


QUESTION = (
    "Are the following Java and Python programs functionally equivalent "
    "(i.e., code clones that solve the same problem)? Answer yes or no."
)


# ── CSV loading + sampling ────────────────────────────────────────────────────

def load_pairs(csv_path, java_map, py_map, n_pos, n_neg):
    """
    Read CSV, resolve file paths, return balanced list of
    (java_path, py_path, label) tuples.
    Each pair is always (java, python) regardless of CSV column order.
    """
    pos, neg = [], []
    needed = max(n_pos, n_neg)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    random.shuffle(rows)

    for row in rows:
        if len(pos) >= needed and len(neg) >= needed:
            break
        i1, i2, lbl = row['index1'], row['index2'], int(row['label'])

        # figure out which is java, which is python
        if i1 in java_map and i2 in py_map:
            jp, pp = java_map[i1], py_map[i2]
        elif i1 in py_map and i2 in java_map:
            jp, pp = java_map[i2], py_map[i1]
        else:
            continue

        entry = (jp, pp, lbl)
        if lbl == 1 and len(pos) < needed:
            pos.append(entry)
        elif lbl == 0 and len(neg) < needed:
            neg.append(entry)

    sample = pos[:n_pos] + neg[:n_neg]
    random.shuffle(sample)
    return sample


# ── writer ────────────────────────────────────────────────────────────────────

def write_jsonl(pairs, out_path):
    written = skipped = 0
    with open(out_path, 'w') as fout:
        for java_path, py_path, label in pairs:
            embs, edges = build_pair_graph(java_path, py_path)
            if embs is None:
                skipped += 1
                continue
            record = {
                'embeddings': embs,
                'edge_index': edges,
                'question': QUESTION,
                'bot': 'yes' if label == 1 else 'no',
                'label': label,
            }
            fout.write(json.dumps(record) + '\n')
            written += 1
    return written, skipped


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_dir',  default=JAVA_DIR)
    parser.add_argument('--py_dir',    default=PY_DIR)
    parser.add_argument('--train_csv', default=TRAIN_CSV)
    parser.add_argument('--test_csv',  default=TEST_CSV)
    parser.add_argument('--out_dir',   default='data_sample/bytecode_clone')
    parser.add_argument('--n_train',   type=int, default=1000)
    parser.add_argument('--n_test',    type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Building index maps...')
    java_map, py_map = build_index_maps(args.java_dir, args.py_dir)
    print(f'  Java files: {len(java_map)}, Py files: {len(py_map)}')

    half_train = args.n_train // 2
    half_test  = args.n_test  // 2

    print(f'Sampling {args.n_train} train pairs ({half_train} pos + {half_train} neg)...')
    train_pairs = load_pairs(args.train_csv, java_map, py_map, half_train, half_train)
    print(f'  Loaded {len(train_pairs)} pairs')

    print(f'Sampling {args.n_test} test pairs ({half_test} pos + {half_test} neg)...')
    test_pairs = load_pairs(args.test_csv, java_map, py_map, half_test, half_test)
    print(f'  Loaded {len(test_pairs)} pairs')

    train_path = os.path.join(args.out_dir, 'train.jsonl')
    test_path  = os.path.join(args.out_dir, 'test.jsonl')

    print(f'Writing {train_path}...')
    w, s = write_jsonl(train_pairs, train_path)
    print(f'  wrote {w}, skipped {s}')

    print(f'Writing {test_path}...')
    w, s = write_jsonl(test_pairs, test_path)
    print(f'  wrote {w}, skipped {s}')

    # quick sanity check
    with open(train_path) as f:
        sample = json.loads(f.readline())
    print(f'\nSanity check (first train sample):')
    print(f'  embeddings: {len(sample["embeddings"])} blocks x {len(sample["embeddings"][0])} dims')
    print(f'  edge_index: {len(sample["edge_index"])} edges')
    print(f'  bot: {sample["bot"]}')
    print('Done.')


if __name__ == '__main__':
    main()
