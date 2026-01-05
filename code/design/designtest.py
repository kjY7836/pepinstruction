import os
import json
import torch
import re
import time
import requests
import gc
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Paths
BASE_MODEL_PATH = "/workspace/contrast/Llama-2-7b-chat-hf"
LORA_ADAPTER_PATH = "/workspace/contrast/merge-model-pronew"
GALACTICA_PATH = "/workspace/contrast/galactica-6.7b"
TEST_DATA_PATH = "design-test.json"

# GPT Config
GPT_API_KEY = os.getenv("CHATANYWHERE_API_KEY", "sk-qrE9NFZK2pAbdZfKONlXbZ69Vgk2WmOWENXdYMqW3kzndW3J")
GPT_API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
GPT_MODEL_NAME = "gpt-3.5-turbo"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 128


if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print("BFloat16 supported. Using bfloat16 to prevent NaNs.")
    COMPUTE_DTYPE = torch.bfloat16
else:
    print("BFloat16 not supported. Falling back to float16.")
    COMPUTE_DTYPE = torch.float16


# 1. Simple Character Alignment (Global)
simple_aligner = PairwiseAligner()
simple_aligner.mode = "global"
simple_aligner.match_score = 1
simple_aligner.mismatch_score = 0
simple_aligner.open_gap_score = 0
simple_aligner.extend_gap_score = 0

# 2. BLOSUM80 Alignment (Short-Peptide Optimized)
# Matches settings from or.py and others: open=-4, extend=-0.1
blosum80 = substitution_matrices.load("BLOSUM80")
blosum_aligner = PairwiseAligner()
blosum_aligner.substitution_matrix = blosum80
blosum_aligner.mode = "global"
blosum_aligner.open_gap_score = -4
blosum_aligner.extend_gap_score = -0.1

def seq_similarity_simple(a: str, b: str) -> float:
    """Character-level global alignment similarity."""
    a, b = a.upper().replace(" ", ""), b.upper().replace(" ", "")
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    score = simple_aligner.score(a, b)
    return score / max(len(a), len(b))

def seq_similarity_blosum(a: str, b: str) -> float:
    """BLOSUM80 global alignment with self-normalization and strict sanitization."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
   
    def sanitize(seq):
        if not seq: return ""
        return "".join([c for c in str(seq).upper() if c in valid_aa])


    a_clean = sanitize(a)
    b_clean = sanitize(b)

    if not a_clean and not b_clean: return 1.0
    if not a_clean or not b_clean: return 0.0
    
    try:

        score = blosum_aligner.score(a_clean, b_clean)
        

        score_a = blosum_aligner.score(a_clean, a_clean)
        score_b = blosum_aligner.score(b_clean, b_clean)
        max_self = (score_a + score_b) / 2
        
        if max_self == 0: return 0.0
        
        sim = score / max_self
        return max(0.0, min(sim, 1.0))
        
    except ValueError as e:
        print(f"[Warning] BLOSUM Alignment failed for input pair. Returning 0.0. Error: {e}")
        return 0.0
    except Exception as e:
        print(f"[Error] Unexpected error in BLOSUM calc: {e}")
        return 0.0



def aggressive_cleanup():
    """Cleans up GPU memory."""
    print("Cleaning up GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    print("GPU memory cleanup done.\n")

def build_prompt(instruction, input_text):
    """
    Standardized prompt for all models (Reference: or.py)
    """
    prefix = (
        "You are a protein design expert. "
        "Given the following protein characteristics, design a realistic amino acid sequence that satisfies all conditions. "
        "The sequence must:\n"
        "- Be between 5 and 100 amino acids long.\n"
        "- Use only standard amino acids (ACDEFGHIKLMNPQRSTVWY).\n"
        "- Avoid unrealistic repetition (e.g., QYQYQYQY, VVVVVV, etc.).\n"
        "- Contain diverse residues with balanced hydrophobic/hydrophilic features.\n"
        "- Output only the sequence without any explanation.\n\n"
        "Now design a sequence for the following input:\n"
    )

    if input_text.strip():
        return f"{prefix}\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        return f"{prefix}\n### Instruction:\n{instruction}\n\n### Response:"

def clean_sequence(text):
    """
    Extracts and cleans amino acid sequence.
    Returns: (raw_extracted, filtered_cleaned)
    """
    text = text.strip()

    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    match = re.search(r"[ACDEFGHIKLMNPQRSTVWY]+", text.upper())
    raw_seq = match.group(0) if match else ""

    filtered_seq = raw_seq
    if len(filtered_seq) > 100:
        filtered_seq = filtered_seq[:100]
 
    filtered_seq = re.sub(r"(..)\1{2,}", r"\1\1", filtered_seq)
    filtered_seq = re.sub(r"(.)\1{4,}", r"\1\1", filtered_seq)
    
    return raw_seq, filtered_seq

def calculate_batch_metrics(instructions, inputs, expected_outputs, generated_texts):
    results = []
    for i, gen_text in enumerate(generated_texts):
        expected = expected_outputs[i].strip()
 
        raw_seq, filtered_seq = clean_sequence(gen_text)
 
        sim_simple_raw = seq_similarity_simple(expected, raw_seq)
        sim_simple_filt = seq_similarity_simple(expected, filtered_seq)
        sim_blosum_raw = seq_similarity_blosum(expected, raw_seq)
        sim_blosum_filt = seq_similarity_blosum(expected, filtered_seq)

        results.append({
            "instruction": instructions[i],
            "input": inputs[i],
            "expected_output": expected,
            "model_output_full": gen_text,
            "generated_sequence_raw": raw_seq,          
            "generated_sequence_filtered": filtered_seq, 
            "metrics": {
                "simple_raw": sim_simple_raw,
                "simple_filtered": sim_simple_filt,
                "blosum_raw": sim_blosum_raw,
                "blosum_filtered": sim_blosum_filt
            }
        })
    return results

def save_model_results(model_name, records, total_time):
    filename_json = f"{model_name}_design_details.json"
    filename_txt = f"{model_name}_design_summary.txt"

    keys = ["simple_raw", "simple_filtered", "blosum_raw", "blosum_filtered"]
    totals = {k: [] for k in keys}
    for r in records:
        for k in keys:
            totals[k].append(r["metrics"][k])
    
    avgs = {k: np.mean(v) if v else 0.0 for k, v in totals.items()}
    
    with open(filename_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Samples: {len(records)}\n")
        f.write(f"Total Time: {total_time:.2f} s\n")
        f.write("-" * 30 + "\n")
        f.write(f"Simple Raw:       {avgs['simple_raw']:.4f}\n")
        f.write(f"Simple Filtered:  {avgs['simple_filtered']:.4f}\n")
        f.write(f"BLOSUM Raw:       {avgs['blosum_raw']:.4f}\n")
        f.write(f"BLOSUM Filtered:  {avgs['blosum_filtered']:.4f}\n")
            
    print(f"Saved results for {model_name}")
    print(f"  Simple (Filt): {avgs['simple_filtered']:.4f}")
    print(f"  BLOSUM (Filt): {avgs['blosum_filtered']:.4f}")


def run_llama_base(data):
    model_name = "Llama-2-7b-Base"
    print(f"\n--- Starting {model_name} Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=COMPUTE_DTYPE,
        device_map=None
    ).to(DEVICE)
    model.eval()
    
    start_time = time.time()
    all_records = []
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        prompts = [build_prompt(item['instruction'], item['input']) for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,      # ENABLED SAMPLING
                temperature=1.0,     # TEMP 1.0
                top_k=50,            # Added top_k for stability with temp=1
                repetition_penalty=1.2
            )
            
        gen_texts = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            gen_texts.append(decoded)
     
            global_idx = i + j
            if global_idx < 2:
                print(f"[DEBUG] {model_name} #{global_idx+1}: {decoded[:100]}...")
            
        batch_res = calculate_batch_metrics(
            [b['instruction'] for b in batch],
            [b['input'] for b in batch],
            [b['output'] for b in batch],
            gen_texts
        )
        all_records.extend(batch_res)
        print(f"Processed batch {i // BATCH_SIZE + 1}")

    save_model_results(model_name, all_records, time.time() - start_time)
    del model, tokenizer
    aggressive_cleanup()

def run_galactica(data):
    model_name = "Galactica-6.7b"
    print(f"\n--- Starting {model_name} Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = AutoTokenizer.from_pretrained(GALACTICA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        GALACTICA_PATH,
        torch_dtype=COMPUTE_DTYPE,
        device_map=None
    ).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    start_time = time.time()
    all_records = []
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        prompts = [build_prompt(item['instruction'], item['input']) for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,      # ENABLED SAMPLING
                temperature=1.0,     # TEMP 1.0
                top_k=50,
                repetition_penalty=1.1, # Matches ga.py slightly
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        gen_texts = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            gen_texts.append(decoded)
            
            if (i + j) < 2:
                print(f"[DEBUG] {model_name} #{(i+j)+1}: {decoded[:100]}...")
            
        batch_res = calculate_batch_metrics(
            [b['instruction'] for b in batch],
            [b['input'] for b in batch],
            [b['output'] for b in batch],
            gen_texts
        )
        all_records.extend(batch_res)
        print(f"Processed batch {i // BATCH_SIZE + 1}")

    save_model_results(model_name, all_records, time.time() - start_time)
    del model, tokenizer
    aggressive_cleanup()

def run_llama_lora(data):
    model_name = "Llama-2-7b-LoRA"
    print(f"\n--- Starting {model_name} Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=COMPUTE_DTYPE,
        device_map=None
    ).to(DEVICE)
    
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.to(DEVICE)
    model.eval()
    
    start_time = time.time()
    all_records = []
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        prompts = [build_prompt(item['instruction'], item['input']) for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,      # ENABLED SAMPLING
                temperature=1.0,     # TEMP 1.0
                top_k=50,
                repetition_penalty=1.2
            )
            
        gen_texts = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            gen_texts.append(decoded)

            if (i + j) < 2:
                print(f"[DEBUG] {model_name} #{(i+j)+1}: {decoded[:100]}...")
            
        batch_res = calculate_batch_metrics(
            [b['instruction'] for b in batch],
            [b['input'] for b in batch],
            [b['output'] for b in batch],
            gen_texts
        )
        all_records.extend(batch_res)
        print(f"Processed batch {i // BATCH_SIZE + 1}")

    save_model_results(model_name, all_records, time.time() - start_time)
    del model, base_model, tokenizer
    aggressive_cleanup()

def run_gpt(data):
    model_name = "GPT-3.5"
    print(f"\n--- Starting {model_name} Evaluation ---")
    start_time = time.time()
    all_records = []
    
    for i, item in enumerate(data):
        prompt = build_prompt(item['instruction'], item['input'])
        
        payload = {
            "model": GPT_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a protein design expert model. Respond precisely with valid amino acid sequences."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 1.0 
        }
        headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
        
        gen_text = ""
        try:
            time.sleep(1) # Rate limit friendly
            response = requests.post(GPT_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
            resp_json = response.json()
            if "choices" in resp_json and resp_json["choices"]:
                gen_text = resp_json["choices"][0]["message"]["content"].strip()
            else:
                print(f"GPT Error on item {i}: {response.text}")
                gen_text = ""
        except Exception as e:
            print(f"GPT Exception on item {i}: {e}")
            gen_text = ""
        
        if i < 2:
             print(f"[DEBUG] {model_name} #{i+1}: {gen_text[:100]}...")

        res = calculate_batch_metrics(
            [item['instruction']], 
            [item['input']], 
            [item['output']], 
            [gen_text]
        )
        all_records.extend(res)
        if (i + 1) % 10 == 0:
            print(f"Processed GPT sample {i + 1}/{len(data)}")
            
    save_model_results(model_name, all_records, time.time() - start_time)


def main():
    print(f"Using Logical Device: {DEVICE}")
    print("(Mapped to Physical GPU 1 via CUDA_VISIBLE_DEVICES)")
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    target_data = [d for d in full_data if d.get("task", "design") == "design"]
    print(f"Total Design Samples: {len(target_data)}")
    
    # Order
    run_llama_base(target_data)
    run_galactica(target_data)
    run_llama_lora(target_data)
    run_gpt(target_data)
    
    print("\nAll evaluations completed successfully.")

if __name__ == "__main__":
    main()
