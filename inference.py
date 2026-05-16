"""
GALLa Inference Script — Java Bug Fix
Usage:
    python3 inference.py --model_path output/instruction/qwen2.5-coder-1.5b/final_step_X
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── prompt format must match training (see data/preprocess_data.py) ──
HUMAN_MARKER = "<s>human\n"
BOT_MARKER   = "<s>bot\n"

def build_prompt(buggy_code: str) -> str:
    human_content = (
        f"# Fix the bug in the following Java program\n\n"
        f"# Buggy\n{buggy_code.strip()}\n\n"
        f"# Fixed\n"
    )
    return f"{HUMAN_MARKER}{human_content}{BOT_MARKER}"


def run_inference(model_path: str, buggy_code: str, max_new_tokens: int = 256) -> str:
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    prompt = build_prompt(buggy_code)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding for deterministic output
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens (skip the prompt)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to fine-tuned model folder (final_step_X)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # example buggy Java program
    buggy_code = """public static TYPE_1 init ( java.lang.String name , java.util.Date date ) {
    TYPE_1 VAR_1 = new TYPE_1 ( ) ;
    VAR_1.METHOD_1( name ) ;
    java.util.Calendar VAR_2 = java.util.Calendar.getInstance ( ) ;
    VAR_2.METHOD_2( date ) ;
    VAR_1.METHOD_3( VAR_2 ) ;
    return VAR_1 ;
}"""

    print("\n=== Buggy Code ===")
    print(buggy_code)

    fixed_code = run_inference(args.model_path, buggy_code, args.max_new_tokens)

    print("\n=== Fixed Code ===")
    print(fixed_code)


if __name__ == "__main__":
    main()
