import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import time
import gc
import requests
import datetime
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score
from collections import defaultdict

# Paths
BASE_MODEL_LLAMA   = "/workspace/contrast/Llama-2-7b-chat-hf"
LORA_ADAPTER_PATH  = "/workspace/contrast/merge-model-pronew"
BASE_MODEL_GAL     = "/workspace/contrast/galactica-6.7b"
DATA_PATH          = "peptide_tasks-test.json"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 10 

# GPT Configuration
GPT_API_KEY = os.getenv("CHATANYWHERE_API_KEY", "sk-qrE9NFZK2pAbdZfKONlXbZ69Vgk2WmOWENXdYMqW3kzndW3J")
GPT_API_URL = os.getenv("CHATANYWHERE_API_URL", "https://api.chatanywhere.tech/v1/chat/completions")
GPT_MODEL = "gpt-3.5-turbo"

# Property Mapping
PROPERTY_ALIASES = {
    "assem": ["assem", "self-assembling peptide"],
    "amp_50": ["amp_50", "antimicrobial peptide"],
    "BBB": ["bbb", "blood-brain barrier penetrating peptide"],
    "CPP": ["cpp", "cell-penetrating peptide"],
    "DPPIV": ["dppiv", "dpp-iv inhibitory peptide"],
    "hemo": ["hemo", "hemolytic peptide"],
    "soluble": ["soluble", "soluble peptide"],
    "toxic": ["toxic", "toxic peptide"]
}


def aggressive_cleanup():
    """Cleans up GPU memory to prevent OOM between model switches."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

def normalize_answer(text: str):
    """Normalize text to 'yes', 'no', or ''."""
    if not text:
        return ""
    text = text.lower().strip()
    text = ''.join([c for c in text if c.isprintable()])
    
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return ""

def build_prompt(instruction, input_text):
    """Construct prompt with strict instruction prefix."""
    prefix = "Answer strictly with one word: Yes or No.\n"
    if input_text and input_text.strip():
        return (
            f"{prefix}Below is an instruction that describes a task, paired with an input. "
            f"Write a response.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        )
    else:
        return (
            f"{prefix}Below is an instruction that describes a task. "
            f"Write a response.\n\n### Instruction:\n{instruction}\n\n### Response:"
        )

def load_and_filter_data():
    """Load and filter data (q&a only, input length <= 200)."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    filtered = []
    skipped_count = 0
    for item in data:
        # Only Q&A tasks
        if item.get("task") != "q&a":
            continue
        inp = item.get("input", "")
        if len(inp) > 200:
            skipped_count += 1
            continue
        filtered.append(item)
    
    print(f"Data Loaded. Total: {len(filtered)}. Skipped (len>200): {skipped_count}.")
    return filtered


def evaluate_predictions(model_name, filename_tag, dataset, predictions):
    """
    Calculate metrics (Accuracy AND Macro-F1) and SAVE to a specific file.
    MODIFIED: Uses Macro-F1 and adds suffix to filename.
    """
    stats_by_prop = defaultdict(lambda: {"true": [], "pred": [], "correct": 0, "total": 0})
    total_correct = 0
    total_count = 0
    
    all_true = []
    all_pred = []
    
    label_map = {"yes": 1, "no": 0}

    for i, item in enumerate(dataset):
        if i >= len(predictions):
            break
            
        raw_output = item.get("output", "")
        gen_output = predictions[i]
        
        expected = normalize_answer(raw_output)
        generated = normalize_answer(gen_output)
        
        # STRICT VALIDITY CHECK: Both must be valid to count
        if expected == "" or generated == "":
            continue
            
        y_true = label_map[expected]
        y_pred = label_map[generated]
        
        is_correct = (y_true == y_pred)
        
        total_count += 1
        if is_correct:
            total_correct += 1
            
        # Collect for overall F1
        all_true.append(y_true)
        all_pred.append(y_pred)
            
        # Property statistics
        instruction = item.get("instruction", "").lower()
        for prop, aliases in PROPERTY_ALIASES.items():
            if any(alias in instruction for alias in aliases):
                stats_by_prop[prop]["total"] += 1
                if is_correct:
                    stats_by_prop[prop]["correct"] += 1
                
                # Collect for per-property F1
                stats_by_prop[prop]["true"].append(y_true)
                stats_by_prop[prop]["pred"].append(y_pred)
                break

    #Construct Output Text 
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Valid Samples (Both exp/gen are Yes/No): {total_count}")
    
    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    lines.append(f"Overall Accuracy: {total_correct}/{total_count} = {overall_acc:.4f}")
    
    #CHANGED: Use Macro Average 
    overall_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    lines.append(f"Overall Macro-F1 Score: {overall_f1:.4f}")
    
    lines.append("-" * 75)
    # Modified Header
    header = f"{'Property':<10} | {'Accuracy':<10} | {'Count':<6} | {'Macro-F1':<10}"
    lines.append(header)
    lines.append("-" * 75)
    
    for prop, stats in stats_by_prop.items():
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        
        # CHANGED: Use Macro Average per property 
        f1 = f1_score(stats["true"], stats["pred"], average='macro', zero_division=0)
        
        row = f"{prop:<10} | {acc:.4f}     | {stats['total']:<6} | {f1:.4f}"
        lines.append(row)
    lines.append("-" * 75)
    
    result_text = "\n".join(lines)

    #  Print to Console
    print(f"\n[{model_name}] Results:")
    print(result_text)

    # Save to File 
    # CHANGED: Added _MacroF1 to filename
    output_filename = f"result_{filename_tag}_MacroF1.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Results saved to {output_filename}")
    
    return overall_acc



def run_local_inference(model, tokenizer, dataset):
    all_generated = []
    total_items = len(dataset)
    total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Starting inference: {total_items} items, {total_batches} batches.")
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, total_items)
        batch = dataset[start:end]
        
        prompts = [build_prompt(item['instruction'], item['input']) for item in batch]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CUTOFF_LEN
        ).to(DEVICE)
        
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
            
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False # Greedy decoding
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            
        for i, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][i])
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            all_generated.append(text)
            
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed batch {batch_idx+1}/{total_batches}")
            
    return all_generated


def run_lora_eval(dataset):
    print("\n" + "="*50)
    print("Model 1: LoRA (Our Model)")
    print("="*50)
    aggressive_cleanup()
    
    # Load Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    # Load Base
    print("Loading Base Model...")
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_LLAMA,
        torch_dtype=torch.float16,
        device_map=None
    ).to(DEVICE)
    
    # Load Adapter
    print("Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).to(DEVICE)
    model.eval()
    model.config.use_cache = False
    
    # Infer
    gen_texts = run_local_inference(model, tokenizer, dataset)
    
    # Evaluate
    evaluate_predictions("LoRA (Our Model)", "LoRA", dataset, gen_texts)
    
    del model, base_model, tokenizer
    aggressive_cleanup()

def run_llama_base_eval(dataset):
    print("\n" + "="*50)
    print("Model 2: Llama-2-7b (Base)")
    print("="*50)
    aggressive_cleanup()
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_LLAMA)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    print("Loading Model...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_LLAMA,
        torch_dtype=torch.float16,
        device_map=None
    ).to(DEVICE)
    model.eval()
    model.config.use_cache = False
    
    gen_texts = run_local_inference(model, tokenizer, dataset)
    
    # Save results
    evaluate_predictions("Llama-2-7b Base", "Llama_Base", dataset, gen_texts)
    
    del model, tokenizer
    aggressive_cleanup()

def run_galactica_eval(dataset):
    print("\n" + "="*50)
    print("Model 3: Galactica-6.7b")
    print("="*50)
    aggressive_cleanup()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_GAL)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_GAL,
        torch_dtype=torch.float16,
        device_map=None
    ).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer)) 
    model.eval()
    model.config.use_cache = False
    
    gen_texts = run_local_inference(model, tokenizer, dataset)
    
    # Save results
    evaluate_predictions("Galactica-6.7b", "Galactica", dataset, gen_texts)
    
    del model, tokenizer
    aggressive_cleanup()

def run_gpt_eval(dataset):
    print("\n" + "="*50)
    print("Model 4: GPT-3.5 (ChatAnywhere)")
    print("="*50)
    
    all_generated = []
    
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"Processing {len(dataset)} items sequentially...")
    
    for i, item in enumerate(dataset):
        prompt_text = build_prompt(item['instruction'], item['input'])
        
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. If the user asks a yes/no question, reply with either 'Yes' or 'No' only."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0
        }
        
        result = ""
        # Retry logic
        for attempt in range(3):
            try:
                resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        result = data["choices"][0]["message"]["content"].strip()
                    break
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        
        all_generated.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"GPT processed {i+1}/{len(dataset)}")
            
        time.sleep(0.1) # Rate limit protection

    # Save results
    evaluate_predictions("GPT-3.5-Turbo", "GPT_3.5", dataset, all_generated)



if __name__ == "__main__":
    print(f"Logical Device: {DEVICE} (Mapped to Physical GPU 1)")
    
    # 1. Load Data
    valid_dataset = load_and_filter_data()
    if len(valid_dataset) == 0:
        print("No valid data found.")
        exit()

    # 2. oreder
    
    try:
        run_lora_eval(valid_dataset)
    except Exception as e:
        print(f"Error running LoRA: {e}")

    try:
        run_llama_base_eval(valid_dataset)
    except Exception as e:
        print(f"Error running Llama Base: {e}")

    try:
        run_galactica_eval(valid_dataset)
    except Exception as e:
        print(f"Error running Galactica: {e}")

    try:
        run_gpt_eval(valid_dataset)
    except Exception as e:
        print(f"Error running GPT: {e}")
        
    print("\nEvaluation Completed.")
