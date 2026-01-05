import os
import json
import torch
import time
import re
import requests
import gc
import random
import numpy as np
from collections import defaultdict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peptidy.descriptors import instability_index, isoelectric_point, charge, x_logp_energy
from Bio.Align import PairwiseAligner

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


TEST_DATA_PATH = "test-new.json"
BASE_MODEL_PATH = "/workspace/contrast/Llama-2-7b-chat-hf"
GALACTICA_PATH = "/workspace/contrast/galactica-6.7b"
LORA_ADAPTER_PATH = "/workspace/contrast/merge-model-pronew"

GPT_API_KEY = os.getenv("CHATANYWHERE_API_KEY", "sk-qrE9NFZK2pAbdZfKONlXbZ69Vgk2WmOWENXdYMqW3kzndW3J")
GPT_API_URL = os.getenv("CHATANYWHERE_API_URL", "https://api.chatanywhere.tech/v1/chat/completions")
GPT_MODEL_NAME = "gpt-3.5-turbo"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 128

SIMILARITY_THRESHOLDS = [0.3, 0.5, 0.7]

aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1
aligner.mismatch_score = 0
aligner.open_gap_score = 0
aligner.extend_gap_score = 0

def calculate_identity(seq1, seq2):
    if not seq1 or not seq2: return 0.0
    score = aligner.score(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    if max_len == 0: return 0.0
    return score / max_len

    def __init__(self):

        self.stats = defaultdict(lambda: {
            'total': 0,
            'valid': 0,        
            'base_success': 0,  
            0.3: 0, 
            0.5: 0, 
            0.7: 0
        })

    def update(self, prop, direction, is_valid, is_improved, identity):
        key = (prop, direction)
        self.stats[key]['total'] += 1
        
        if is_valid:
            self.stats[key]['valid'] += 1
        
        if is_improved:
            self.stats[key]['base_success'] += 1
            for t in SIMILARITY_THRESHOLDS:
                if identity > t:
                    self.stats[key][t] += 1

    def get_summary(self):
        summary = {}
   
        grand_total = sum(v['total'] for v in self.stats.values())
        grand_valid = sum(v['valid'] for v in self.stats.values())
        grand_base = sum(v['base_success'] for v in self.stats.values())
        grand_thresholds = {t: sum(v[t] for v in self.stats.values()) for t in SIMILARITY_THRESHOLDS}
        
        summary['Overall'] = {
            'total': grand_total,
            'valid_rate': grand_valid / grand_total if grand_total > 0 else 0,
            'base_rate': grand_base / grand_total if grand_total > 0 else 0,
            'threshold_rates': {t: grand_thresholds[t] / grand_total if grand_total > 0 else 0 for t in SIMILARITY_THRESHOLDS}
        }


        for key, val in self.stats.items():
            prop, direction = key
            total = val['total']
            if total == 0: continue
            
            task_name = f"{prop} ({direction})"
            summary[task_name] = {
                'total': total,
                'valid_rate': val['valid'] / total,
                'base_rate': val['base_success'] / total,
                'threshold_rates': {t: val[t] / total for t in SIMILARITY_THRESHOLDS}
            }
        return summary


def build_prompt_standard(instruction, input_text):
    """用于 Llama Base, LoRA 和 GPT"""
    prefix = (
        "Make sure the modified amino acid sequence is not identical to the input sequence, even if only one residue is changed."
        "When answering, first provide a specific amino acid sequence that is different from the original amino acid sequence.\n"
    )
    return f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"

def build_prompt_galactica(instruction, input_text):
    """用于 Galactica (专属优化)"""
    prefix = ("You are a protein engineer. Modify the amino acid sequence to optimize "
              "the target property. First output a specific improved amino acid sequence.\n")
    return f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

def extract_sequence(text, input_seq):
    if not text: return ""
    aa_regex = r"[ACDEFGHIKLMNPQRSTVWY]+"

    matches = re.findall(r"Modified sequence:\s*(" + aa_regex + ")", text)
    if not matches:
 
        matches = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{3,}", text)
    
    for seq in matches:
        clean_seq = seq.strip()
   
        if clean_seq != input_seq:
            return clean_seq
    return ""

def extract_property_info(item):
    text = item["instruction"]
    props = ["instability_index", "isoelectric_point", "charge", "x_logp_energy"]
    target_prop = None
    
    for p in props:
        if p in text.lower() or (item.get("output") and p in item["output"].lower()):
            target_prop = p
            break
            
    direction = item.get("task", "")
    if not direction:
        if "increase" in text.lower() or "higher" in text.lower():
            direction = "increase"
        elif "decrease" in text.lower() or "lower" in text.lower():
            direction = "decrease"
        elif target_prop == "instability_index":
            direction = "decrease"
    
    return target_prop, direction

def compute_prop(seq, target_property):
    try:
        if not seq: return None
        if target_property == "instability_index": return float(instability_index(seq))
        elif target_property == "isoelectric_point": return float(isoelectric_point(seq))
        elif target_property == "charge": return float(charge(seq))
        elif target_property == "x_logp_energy": return float(x_logp_energy(seq))
        else: return None
    except: return None

def check_property_improvement(prop_in, prop_out, target_property, direction):
    if prop_in is None or prop_out is None: return False
    if target_property == "instability_index" and not direction:
        return prop_out < prop_in  # Default
    if direction == "increase":
        return prop_out > prop_in
    elif direction == "decrease":
        return prop_out < prop_in
    return False

def format_summary_string(model_name, summary_data, total_time):
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"Total Time: {total_time:.2f} s")
    lines.append("-" * 100)

    lines.append(f"{'Task':<35} | {'Count':<5} | {'Valid%':<7} | {'Base%':<7} | {'>30%':<7} | {'>50%':<7} | {'>70%':<7}")
    lines.append("-" * 100)
    
    keys = sorted([k for k in summary_data.keys() if k != 'Overall'])
    if 'Overall' in summary_data:
        keys.insert(0, 'Overall')
        
    for k in keys:
        d = summary_data[k]
        row = (f"{k:<35} | {d['total']:<5} | "
               f"{d['valid_rate']:.1%} | "
               f"{d['base_rate']:.1%} | "
               f"{d['threshold_rates'][0.3]:.1%} | "
               f"{d['threshold_rates'][0.5]:.1%} | "
               f"{d['threshold_rates'][0.7]:.1%}")
        lines.append(row)
    lines.append("-" * 100)
    return "\n".join(lines)

def save_result_file(model_name, results, summary_data, total_time):
    slug = model_name.lower().replace(" ", "-")
    with open(f"{slug}-eval-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    summary_text = format_summary_string(model_name, summary_data, total_time)
    with open(f"{slug}-eval-summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n{summary_text}\n")
    print(f"Results saved for {model_name}")



def evaluate_gpt(test_data):
    model_name = "GPT-3.5-Turbo"
    print(f"\nStarting evaluation for {model_name}...")
    
    tracker = MetricTracker()
    results = []
    start_time = time.time()
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    
    for i, item in enumerate(test_data):
        target_prop, direction = extract_property_info(item)
        if not target_prop: continue


        prompt = build_prompt_standard(item["instruction"], item["input"])
        
        gen_text = ""
        for _ in range(3):
            try:
                payload = {
                    "model": GPT_MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0, "max_tokens": 512
                }
                resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    gen_text = resp.json()["choices"][0]["message"]["content"].strip()
                    break
                time.sleep(1)
            except: time.sleep(1)
        
        input_seq = item["input"]
        gen_seq = extract_sequence(gen_text, input_seq)
        

        is_valid = bool(gen_seq)
        
        prop_in = compute_prop(input_seq, target_prop)
        prop_out = compute_prop(gen_seq, target_prop)
        is_improved = check_property_improvement(prop_in, prop_out, target_prop, direction)
        identity = calculate_identity(input_seq, gen_seq)
        
        tracker.update(target_prop, direction, is_valid, is_improved, identity)
        results.append({
            "task": f"{target_prop} ({direction})", "input": input_seq, "output": gen_seq,
            "is_valid": is_valid, "is_improved": is_improved, "similarity": identity
        })
        if (i + 1) % 10 == 0: print(f"GPT Progress: {i+1}/{len(test_data)}")
            
    total_time = time.time() - start_time
    summary = tracker.get_summary()
    save_result_file(model_name, results, summary, total_time)
    return summary

def evaluate_local_model(model_name, model, tokenizer, test_data, prompt_fn):
    print(f"\nStarting evaluation for {model_name} on {DEVICE}...")
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "repetition_penalty": 1.2,
    }
    
    tracker = MetricTracker()
    results = []
    start_time = time.time()
    total_count = len(test_data)
    
    for i in range(0, total_count, BATCH_SIZE):
        batch = test_data[i : i + BATCH_SIZE]

        prompts = [prompt_fn(item["instruction"], item["input"]) for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN)
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            
        for j, out_tokens in enumerate(outputs):
            item = batch[j]
            target_prop, direction = extract_property_info(item)
            if not target_prop: continue

            input_len = len(inputs["input_ids"][j])
            gen_text = tokenizer.decode(out_tokens[input_len:], skip_special_tokens=True).strip()
            
            input_seq = item["input"]
            gen_seq = extract_sequence(gen_text, input_seq)
            

            is_valid = bool(gen_seq)
            
            prop_in = compute_prop(input_seq, target_prop)
            prop_out = compute_prop(gen_seq, target_prop)
            
            is_improved = check_property_improvement(prop_in, prop_out, target_prop, direction)
            identity = calculate_identity(input_seq, gen_seq)
            
            tracker.update(target_prop, direction, is_valid, is_improved, identity)
            
            results.append({
                "task": f"{target_prop} ({direction})",
                "input": input_seq, "output": gen_seq, "raw_text": gen_text,
                "is_valid": is_valid, "is_improved": is_improved, "similarity": identity
            })
            
        print(f"Batch {i//BATCH_SIZE + 1} processed ({min(i+BATCH_SIZE, total_count)}/{total_count})")
        
    total_time = time.time() - start_time
    summary = tracker.get_summary()
    save_result_file(model_name, results, summary, total_time)
    return summary


def run_galactica(test_data):
    """顺序1: Galactica (专属 Prompt)"""
    cleanup()
    print("Loading Galactica...")
    tokenizer = AutoTokenizer.from_pretrained(GALACTICA_PATH)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        GALACTICA_PATH,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model.config.use_cache = False
    

    return evaluate_local_model("Galactica", model, tokenizer, test_data, prompt_fn=build_prompt_galactica)

def run_llama_base(test_data):
    """顺序2: Llama Base (标准 Prompt)"""
    cleanup()
    print("Loading Llama Base...")
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": 0} 
    )
    model.eval()
    model.config.use_cache = False
    return evaluate_local_model("Llama-Base", model, tokenizer, test_data, prompt_fn=build_prompt_standard)

def run_llama_lora(test_data):
    """顺序3: Llama LoRA (标准 Prompt)"""
    cleanup()
    print("Loading Llama LoRA...")
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = False
    return evaluate_local_model("Llama-LoRA", model, tokenizer, test_data, prompt_fn=build_prompt_standard)


def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} samples.")

 
    
    # 1. Galactica 
    gal_stats = run_galactica(test_data)

    # 2. Llama Base
    base_stats = run_llama_base(test_data)

    # 3. Llama LoRA
    lora_stats = run_llama_lora(test_data)

    # 4. GPT-3.5
    gpt_stats = evaluate_gpt(test_data)

    with open("final_comparison_all_tasks.txt", "w", encoding="utf-8") as f:
        f.write("========== FINAL COMPARISON ==========\n\n")
        
        models = [
            ("Galactica", gal_stats),
            ("Llama Base", base_stats),
            ("Llama LoRA", lora_stats),
            ("GPT-3.5", gpt_stats)
        ]
        
        for name, stats in models:
            summary_str = format_summary_string(name, stats, 0)
            f.write(summary_str + "\n\n")

    print("\nAll Done! Check 'final_comparison_all_tasks.txt' for details.")

if __name__ == "__main__":
    main()

