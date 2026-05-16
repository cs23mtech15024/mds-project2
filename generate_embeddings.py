"""
GALLa Embedding Generation — Stage 1 model
Extracts graph-aware code embeddings from the Stage 1 checkpoint.

Usage:
    python3 generate_embeddings.py --model_path output/test/qwen2.5-coder-1.5b/final_step_X
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_embedding(model, tokenizer, code: str, device: str) -> np.ndarray:
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last hidden state: shape [1, seq_len, hidden_dim]
    last_hidden = outputs.hidden_states[-1]

    # mean pooling over token dimension → shape [hidden_dim]
    embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to Stage 1 checkpoint folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # example Java code snippets
    code_samples = [
        """public static int add(int a, int b) {
    return a + b;
}""",
        """public static int multiply(int a, int b) {
    return a * b;
}""",
        """public void printList(List<String> items) {
    for (String item : items) {
        System.out.println(item);
    }
}""",
    ]

    print(f"\nGenerating embeddings for {len(code_samples)} code samples...\n")
    embeddings = []
    for i, code in enumerate(code_samples):
        emb = get_embedding(model, tokenizer, code, device)
        embeddings.append(emb)
        print(f"Sample {i+1}: embedding shape = {emb.shape}")

    embeddings = np.array(embeddings)
    print(f"\nAll embeddings shape: {embeddings.shape}")

    # example: compute cosine similarity between sample 1 and 2
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nCosine similarity (add vs multiply): {cosine_sim(embeddings[0], embeddings[1]):.4f}")
    print(f"Cosine similarity (add vs printList): {cosine_sim(embeddings[0], embeddings[2]):.4f}")

    # save embeddings to file
    np.save("code_embeddings.npy", embeddings)
    print(f"\nEmbeddings saved to code_embeddings.npy")


if __name__ == "__main__":
    main()
