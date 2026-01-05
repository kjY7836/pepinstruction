import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import re
import pandas as pd
import time
import requests
import gc
from difflib import SequenceMatcher
from collections import defaultdict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL_LLAMA = "/workspace/contrast/Llama-2-7b-chat-hf"
BASE_MODEL_GALACTICA = "/workspace/contrast/galactica-6.7b"
LORA_MODEL_PATH_GEN = "/workspace/contrast/merge-model-pronew"   # LoRA for generation
LORA_MODEL_PATH_QA = "/workspace/contrast/qa"                    # LoRA for QA
CSV_FILE = "peptides_filtered.csv"
DATA_FILE = "peptide_tasks-upmore-1000-modified.json"

# GPT Config
CHATANYWHERE_URL = "https://api.chatanywhere.tech/v1/chat/completions"
CHATANYWHERE_KEY = "sk-qrE9NFZK2pAbdZfKONlXbZ69Vgk2WmOWENXdYMqW3kzndW3J"
CHATANYWHERE_MODEL = "gpt-3.5-turbo"

# Parameters
CUTOFF_LEN = 512
DEVICE = torch.device("cuda:0") 

# Output Directories
OUTPUT_BASE_DIR = "./1027_merged"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


def aggressive_cleanup():
    """Aggressively clean GPU memory to prevent OOM."""
    print("[INFO] Cleaning up GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print("[INFO] Cleanup done.")

def normalize_answer(text: str):
    """Normalize QA model output to yes/no."""
    if text is None:
        return ""
    text = text.lower().strip()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return ""

def extract_sequence(text, input_seq):
    """
    Extract amino acid sequence with context filtering.
    Removes property names and unrelated text before extraction.
    """
    if not text:
        return ""
    
    # Context Filtering: Remove common property names that interfere with extraction
    text_filtered = re.sub(r'\b(DPP-IV|DPP IV|DPP)\b', '', text, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bself[-\s]?assembl(ing|y)\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bantimicrobial\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bblood[-\s]?brain[-\s]?barrier\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bcell[-\s]?penetrating\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bhemolytic\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bsoluble\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\btoxic\b', '', text_filtered, flags=re.IGNORECASE)
    text_filtered = re.sub(r'\bpeptide\b', '', text_filtered, flags=re.IGNORECASE)
    
    # Extract sequence candidates
    matches = re.findall(r"Modified sequence:\s*([ACDEFGHIKLMNPQRSTVWY]+)", text_filtered)
    if not matches:
        matches = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{3,}", text_filtered)
    
    for seq in matches:
        # Return the first one that isn't the input itself
        if seq != input_seq:
            return seq
    return ""

def extract_property_and_label(instruction):
    """Parse instruction to identify property, target label (0/1), and direction."""
    instr_lower = instruction.lower()
    
    if re.search(r"dpp[-\s]?iv", instr_lower):
        prop = "DPP-IV inhibitory peptide"
    elif "self-assembling" in instr_lower:
        prop = "self-assembling peptide"
    elif "antimicrobial" in instr_lower:
        prop = "antimicrobial peptide"
    elif "blood-brain barrier" in instr_lower:
        prop = "blood-brain barrier penetrating peptide"
    elif "cell-penetrating" in instr_lower:
        prop = "cell-penetrating peptide"
    elif "hemolytic" in instr_lower:
        prop = "hemolytic peptide"
    elif "soluble" in instr_lower:
        prop = "soluble peptide"
    elif "toxic" in instr_lower:
        prop = "toxic peptide"
    else:
        prop = None

    if "lose" in instr_lower:
        label = 0
        direction = "lose"
    elif "gain" in instr_lower:
        label = 1
        direction = "gain"
    else:
        label = None
        direction = None

    return prop, label, direction

def find_similar_sequences(df_peptides, input_seq, property_name, target_label, top_k=3):
    """Find few-shot examples."""
    candidates = df_peptides[(df_peptides["property"] == property_name) &
                             (df_peptides["label"] == target_label)]
    if candidates.empty:
        return []
    scored = [(seq, SequenceMatcher(None, input_seq, seq).ratio()) for seq in candidates["sequence"]]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in scored[:top_k]]



def build_prompt_gen(instruction, input_text, ref_seq=None):
    prefix = ("You are a protein engineer. Modify the amino acid sequence to optimize "
              "the target property. First output a specific improved amino acid sequence.\n")
    if ref_seq:
        return (f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n"
                f"Here is an example peptide with the target property: {ref_seq}\n\n### Response:")
    else:
        return f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"

def build_prompt_qa(property_name, seq):
    prefix = "You are a protein property predictor. Answer strictly Yes or No.\n"
    return f"{prefix}### Question:\nIs the following amino acid sequence a {property_name}?\n\n### Sequence:\n{seq}\n\n### Response:"



def local_infer(model, tokenizer, prompts, max_new_tokens=128):
    """Generic inference for local models."""
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    gen_texts = []
    for i, out in enumerate(outputs):
        # Slicing input to return only generated tokens
        input_len = len(inputs["input_ids"][i])
        gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        gen_texts.append(gen_text)
    return gen_texts

def gpt_infer(prompt, retries=3):
    """API inference for ChatAnywhere."""
    payload = {
        "model": CHATANYWHERE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a protein design assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {CHATANYWHERE_KEY}",
        "Content-Type": "application/json"
    }
    for attempt in range(retries):
        try:
            response = requests.post(CHATANYWHERE_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                return content
        except Exception as e:
            time.sleep(2)
    return ""



def run_evaluation_loop(model_name, sub_dir, data, df_peptides, gen_func, qa_func):
    """
    Main Logic Loop.
    Logic Update:
    - If model_name == "LoRA": max_rounds = 3 (Retry enabled)
    - If model_name != "LoRA": max_rounds = 1 (One-shot only)
    """
    print(f"\n[INFO] Starting evaluation for: {model_name}")
    output_dir = os.path.join(OUTPUT_BASE_DIR, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    save_file = os.path.join(output_dir, "ret-merged.jsonl")
    stats_file = os.path.join(output_dir, "success_stats.json")
    sim_stats_file = os.path.join(output_dir, "similarity_success_stats.json")


    if model_name == "LoRA":
        max_rounds = 3
        print("[CONFIG] Mode: 3-round retry enabled (Our Model).")
    else:
        max_rounds = 1
        print(f"[CONFIG] Mode: 1-round only (Baseline: {model_name}).")


    success_total = 0
    success_first = 0
    similarity_scores = []
    
    # Stats containers
    property_stats = defaultdict(lambda: {"success_first": 0, "success_total": 0, "total": 0, "similarities": []})
    
    similarity_ranges = ["30%+", "50%+", "75%+"]
    sim_stats = defaultdict(
        lambda: defaultdict(
            lambda: {"total_samples": 0, "success_in_ranges": {r: 0 for r in similarity_ranges}}
        )
    )

    start_time = time.time()

    for idx, item in enumerate(data, 1):
        instruction = item["instruction"]
        input_seq = item["input"]
        
        prop, target_label, direction = extract_property_and_label(instruction)
        if prop is None: prop = "Unknown"
        if direction is None: direction = "Unknown"

        # Find n-shot references
        ref_candidates = find_similar_sequences(df_peptides, input_seq, prop, target_label, top_k=3)
        
        gen_seq = ""
        first_gen_seq = None
        success = False
        attempt_used = 0
        final_ref_seq = None
        
        # --- Retry Loop (Controlled by max_rounds) ---
        for attempt in range(max_rounds):
            ref_seq = ref_candidates[attempt] if attempt < len(ref_candidates) else None
            
            # 1. Generate
            prompt_gen = build_prompt_gen(instruction, input_seq, ref_seq)
            gen_text = gen_func(prompt_gen) # Call abstract generator
            current_gen_seq = extract_sequence(gen_text, input_seq)
            
            if not current_gen_seq:
                continue
            
            if attempt == 0:
                first_gen_seq = current_gen_seq
            
            gen_seq = current_gen_seq
            
            # 2. QA Validate
            prompt_qa = build_prompt_qa(prop, gen_seq)
            qa_text = qa_func(prompt_qa) # Call abstract QA
            qa_norm = normalize_answer(qa_text)
            
            pred_label = 1 if qa_norm == "yes" else 0 if qa_norm == "no" else None
            
            if pred_label is not None and pred_label == target_label:
                success = True
                attempt_used = attempt + 1
                final_ref_seq = ref_seq
                break
        
        if not success:
            attempt_used = max_rounds
            final_ref_seq = ref_candidates[0] if ref_candidates else None
            if not first_gen_seq and gen_seq:
                first_gen_seq = gen_seq
        
        # --- Statistics ---
        sim_score = None
        if first_gen_seq:
            sim_score = SequenceMatcher(None, input_seq, first_gen_seq).ratio()
            similarity_scores.append(sim_score)
            
        if success:
            success_total += 1
            if attempt_used == 1:
                success_first += 1
        
        # Update Property Stats
        property_stats[prop]["total"] += 1
        property_stats[prop]["similarities"].append(sim_score)
        if success:
            property_stats[prop]["success_total"] += 1
            if attempt_used == 1:
                property_stats[prop]["success_first"] += 1
        
        # Update Similarity Group Stats
        sim_stats[prop][direction]["total_samples"] += 1
        if success and attempt_used == 1 and sim_score is not None:
            if sim_score >= 0.75:
                sim_stats[prop][direction]["success_in_ranges"]["75%+"] += 1
            if sim_score >= 0.5:
                sim_stats[prop][direction]["success_in_ranges"]["50%+"] += 1
            if sim_score >= 0.3:
                sim_stats[prop][direction]["success_in_ranges"]["30%+"] += 1
        
        # Save Result
        result_record = {
            "instruction": instruction,
            "input": input_seq,
            "output": gen_seq,
            "success": success,
            "attempts_used": attempt_used,
            "property": prop,
            "ref_seq": final_ref_seq,
            "similarity_first_gen": sim_score,
            "model": model_name
        }
        with open(save_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
        if idx % 50 == 0:
            print(f"[{model_name}] Processed {idx}/{len(data)} - Success: {success_total}")

    # --- Save Summary ---
    total_tasks = len(data)
    avg_sim = sum([s for s in similarity_scores if s]) / len([s for s in similarity_scores if s]) if similarity_scores else 0
    
    summary = {
        "model": model_name,
        "total_tasks": total_tasks,
        "success_rate_1shot": success_first / total_tasks if total_tasks else 0,
        "success_rate_3shot": success_total / total_tasks if total_tasks else 0,
        "avg_similarity_first": avg_sim,
        "property_stats": {
            k: {
                "success_rate_1shot": v["success_first"]/v["total"] if v["total"] else 0,
                "success_rate_3shot": v["success_total"]/v["total"] if v["total"] else 0,
                "count": v["total"]
            } for k, v in property_stats.items()
        }
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    sim_summary = {}
    for p, d_stats in sim_stats.items():
        sim_summary[p] = {}
        for d, s in d_stats.items():
            tot = s["total_samples"]
            ranges_res = {}
            for r in similarity_ranges:
                cnt = s["success_in_ranges"][r]
                rate = (cnt / tot * 100) if tot > 0 else 0.0
                ranges_res[r] = {
                    "success_count": cnt,
                    "total_samples": tot,
                    "success_rate(%)": round(rate, 2)
                }
            sim_summary[p][d] = ranges_res
            
    with open(sim_stats_file, "w", encoding="utf-8") as f:
        json.dump(sim_summary, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"[DONE] {model_name} Finished. Time: {elapsed/60:.1f} min")



def main():
    print(f"[INFO] Using Device: {DEVICE}")
    print("[INFO] (This is mapped to Physical GPU 1)")

    # Load Data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df_peptides = pd.read_csv(CSV_FILE)

    # STAGE 1: LoRA Model (3 Rounds)

    print("\n" + "="*40)
    print("   STAGE 1: LoRA Model (Our Model)")
    print("="*40)
    aggressive_cleanup()

    try:
        print("[INFO] Loading Shared Base Model (Llama-2-7b)...")
        # Load ONE base model
        tokenizer_shared = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
        tokenizer_shared.pad_token_id = 0
        tokenizer_shared.padding_side = "left"
        
        model_shared = LlamaForCausalLM.from_pretrained(
            BASE_MODEL_LLAMA, torch_dtype=torch.float16, device_map={"": DEVICE}
        )

        print("[INFO] Loading 'gen' adapter...")
        model_shared = PeftModel.from_pretrained(
            model_shared, LORA_MODEL_PATH_GEN, adapter_name="gen"
        )
        
        print("[INFO] Loading 'qa' adapter...")
        model_shared.load_adapter(LORA_MODEL_PATH_QA, adapter_name="qa")
        
        model_shared.eval()
        model_shared.config.use_cache = False

        def lora_gen(prompt):
            model_shared.set_adapter("gen")
            return local_infer(model_shared, tokenizer_shared, [prompt])[0]
        
        def lora_qa(prompt):
            model_shared.set_adapter("qa")
            return local_infer(model_shared, tokenizer_shared, [prompt], max_new_tokens=16)[0]

        run_evaluation_loop("LoRA", "lora", data, df_peptides, lora_gen, lora_qa)
        
        del model_shared, tokenizer_shared
        aggressive_cleanup()

    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}")
        aggressive_cleanup()


    # STAGE 2:Base Model (1 Round)

    print("\n" + "="*40)
    print("   STAGE 2: Original (Base) Model")
    print("="*40)
    
    try:
        print("[INFO] Loading Shared Base Model...")
        tokenizer_shared = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
        tokenizer_shared.pad_token_id = 0
        tokenizer_shared.padding_side = "left"
        
        model_shared = LlamaForCausalLM.from_pretrained(
            BASE_MODEL_LLAMA, torch_dtype=torch.float16, device_map={"": DEVICE}
        )

        # Only load QA adapter. For generation, we disable adapter.
        print("[INFO] Loading 'qa' adapter...")
        model_shared = PeftModel.from_pretrained(
            model_shared, LORA_MODEL_PATH_QA, adapter_name="qa"
        )
        model_shared.eval()
        model_shared.config.use_cache = False

        def or_gen(prompt):
            # Disable adapter to use Base Model weights
            with model_shared.disable_adapter():
                return local_infer(model_shared, tokenizer_shared, [prompt])[0]
        
        def or_qa(prompt):
            model_shared.set_adapter("qa")
            return local_infer(model_shared, tokenizer_shared, [prompt], max_new_tokens=16)[0]

        run_evaluation_loop("Original", "or", data, df_peptides, or_gen, or_qa)

        del model_shared, tokenizer_shared
        aggressive_cleanup()

    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        aggressive_cleanup()

  
    # STAGE 3: Galactica (1 Round)

    print("\n" + "="*40)
    print("   STAGE 3: Galactica")
    print("="*40)

    try:
        # Load Galactica
        print("[INFO] Loading Galactica...")
        tokenizer_ga = AutoTokenizer.from_pretrained(BASE_MODEL_GALACTICA)
        if tokenizer_ga.pad_token is None:
            tokenizer_ga.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer_ga.padding_side = "left"
        
        model_ga = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_GALACTICA, torch_dtype=torch.float16, device_map={"": DEVICE}
        )
        model_ga.resize_token_embeddings(len(tokenizer_ga))
        model_ga.eval()

        # Load QA
        print("[INFO] Loading QA Model for Discriminator...")
        base_qa = LlamaForCausalLM.from_pretrained(
            BASE_MODEL_LLAMA, torch_dtype=torch.float16, device_map={"": DEVICE}
        )
        model_qa = PeftModel.from_pretrained(base_qa, LORA_MODEL_PATH_QA, torch_dtype=torch.float16)
        model_qa.eval()
        model_qa.config.use_cache = False
        tokenizer_qa = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
        tokenizer_qa.pad_token_id = 0
        tokenizer_qa.padding_side = "left"

        def ga_gen(prompt):
            inputs = tokenizer_ga([prompt], return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
            if 'token_type_ids' in inputs: del inputs['token_type_ids']
            with torch.no_grad():
                outputs = model_ga.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2, pad_token_id=tokenizer_ga.pad_token_id)
            input_len = len(inputs["input_ids"][0])
            return tokenizer_ga.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        def ga_qa(prompt):
            return local_infer(model_qa, tokenizer_qa, [prompt], max_new_tokens=16)[0]

        run_evaluation_loop("Galactica", "ga", data, df_peptides, ga_gen, ga_qa)

        del model_ga, tokenizer_ga, model_qa, tokenizer_qa, base_qa
        aggressive_cleanup()

    except Exception as e:
        print(f"[ERROR] Stage 3 failed: {e}")
        aggressive_cleanup()

    # STAGE 4: GPT-3.5 (1 Round)
    print("\n" + "="*40)
    print("   STAGE 4: GPT-3.5")
    print("="*40)

    try:
        # Only need QA model loaded locally
        print("[INFO] Loading QA Model for Discriminator...")
        base_qa = LlamaForCausalLM.from_pretrained(
            BASE_MODEL_LLAMA, torch_dtype=torch.float16, device_map={"": DEVICE}
        )
        model_qa = PeftModel.from_pretrained(base_qa, LORA_MODEL_PATH_QA, torch_dtype=torch.float16)
        model_qa.eval()
        model_qa.config.use_cache = False
        tokenizer_qa = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
        tokenizer_qa.pad_token_id = 0
        tokenizer_qa.padding_side = "left"

        def gpt_qa_func(prompt):
            return local_infer(model_qa, tokenizer_qa, [prompt], max_new_tokens=16)[0]

        run_evaluation_loop("GPT-3.5", "gpt", data, df_peptides, gpt_infer, gpt_qa_func)

        del model_qa, tokenizer_qa, base_qa
        aggressive_cleanup()

    except Exception as e:
        print(f"[ERROR] Stage 4 failed: {e}")
        aggressive_cleanup()

    print("\n[SUCCESS] All evaluation stages completed.")

if __name__ == "__main__":
    main()
