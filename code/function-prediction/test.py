import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import nltk
import re
import time
import requests
import gc
import numpy as np
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

# Paths
BASE_MODEL_PATH = "/workspace/contrast/Llama-2-7b-chat-hf"
LORA_ADAPTER_PATH = "/workspace/contrast/merge-model-pronew"
GALACTICA_PATH = "/workspace/contrast/galactica-6.7b"
TEST_DATA_PATH = "fc-test_modified.json"

# GPT Config
GPT_API_KEY = "sk-qrE9NFZK2pAbdZfKONlXbZ69Vgk2WmOWENXdYMqW3kzndW3J"
GPT_API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
GPT_MODEL_NAME = "gpt-3.5-turbo"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
CUTOFF_LEN = 512
MAX_NEW_TOKENS = 128

# NLTK Setup
nltk_data_dir = os.path.expanduser("~/nltk_data")
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method4


def aggressive_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    print("GPU memory cleanup done.\n")

def normalize_text(text: str) -> str:
    if text is None: return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_expected_keywords(text: str):
    if not text: return []
    parts = []
    m1 = re.findall(r"The protein probably be able to ([^.]+)\.", text, flags=re.IGNORECASE)
    m2 = re.findall(r"The protein probably be able to regulate ([^.]+)\.", text, flags=re.IGNORECASE)
    m3 = re.findall(r"The protein localizes to the ([^.]+)\.", text, flags=re.IGNORECASE)
    if m1: parts.extend(m1)
    if m2: parts.extend(m2)
    if m3: parts.extend(m3)
    if not parts:
        parts = re.split(r"[;,]", text)
    kws = []
    for p in parts:
        for s in re.split(r";", p):
            s2 = s.strip()
            if s2:
                s2 = re.sub(r"[\.]+$", "", s2).strip()
                if s2: kws.append(s2)
    return kws

def build_vocab(data):
    vocab_set = set()
    for item in data:
        if item.get("task") != "function-prediction": continue
        candidates = extract_expected_keywords(item.get("output", ""))
        for c in candidates:
            norm = normalize_text(c)
            if norm: vocab_set.add(norm)
    return sorted(list(vocab_set))

def extract_with_vocab(text, vocab):
    if not text: return []
    text_norm = normalize_text(text)
    found = []
    for term in vocab:
        if term in text_norm: found.append(term)
    return sorted(found)

def calculate_keyword_metrics(expected_text, generated_text, vocab):
    exp_norm = []
    candidates = extract_expected_keywords(expected_text)
    for c in candidates:
        cc = normalize_text(c)
        if cc in vocab: exp_norm.append(cc)
    pred_norm = extract_with_vocab(generated_text, vocab)
    exp_set = set(exp_norm)
    pred_set = set(pred_norm)
    if not exp_set and not pred_set: return 0.0
    if not pred_set: return 0.0
    inter = exp_set & pred_set
    precision = len(inter) / len(pred_set)
    recall = len(inter) / len(exp_set) if exp_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1

def build_prompt(instruction, input_text):
    sys_msg = "You are a protein biologist specializing in function prediction. Analyze the protein sequence and describe its predicted functions, biological processes, and cellular locations using standardized Gene Ontology (GO) terms."
    if input_text.strip():
        return f"{sys_msg}\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        return f"{sys_msg}\nBelow is an instruction that describes a task. Write a response.\n\n### Instruction:\n{instruction}\n\n### Response:"

def clean_output(output_text, prompt_text=""):
    if prompt_text and output_text.startswith(prompt_text):
        output_text = output_text[len(prompt_text):]
    stop_tokens = ["<eos>", "</s>", "###", "[PAD]"]
    for token in stop_tokens:
        if token in output_text: output_text = output_text.split(token)[0]
    return output_text.strip()

def save_results(model_name, records, avg_metrics, total_time):
    with open(f"{model_name}_details.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(f"{model_name}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nTotal Samples: {len(records)}\nTotal Time: {total_time:.2f} s\nDevice: {DEVICE}\n" + "-"*40 + "\n")
        for k, v in avg_metrics.items(): f.write(f"{k}: {v:.4f}\n")
    print(f"Results saved for {model_name}")

def evaluate_batch_logic(prompts, instructions, inputs, expecteds, vocab, infer_func):
    generated_texts = infer_func(prompts)
    batch_records = []
    for i, gen_text in enumerate(generated_texts):
        expected = expecteds[i]
        gen_clean = clean_output(gen_text, prompts[i]).replace("#", "").replace("##", "").strip()
        
        # ROUGE-L
        try: r_score = scorer.score(expected, gen_clean)["rougeL"].fmeasure
        except: r_score = 0.0
        
        # BLEU
        try: b_score = sentence_bleu([word_tokenize(expected)], word_tokenize(gen_clean), smoothing_function=smoother)
        except: b_score = 0.0
        
        # F1 Only
        f1 = calculate_keyword_metrics(expected, gen_clean, vocab)
        
        batch_records.append({
            "instruction": instructions[i], "input": inputs[i], "expected": expected, "generated": gen_clean,
            "metrics": {"ROUGE-L": r_score, "BLEU": b_score, "F1": f1}
        })
    return batch_records

def process_dataset(data, vocab, infer_fn, model_name, batch_size=BATCH_SIZE):
    start_time = time.time()
    target_data = [d for d in data if d.get("task") == "function-prediction"]
    total_items = len(target_data)
    print(f"Total samples for {model_name}: {total_items}")
    all_records = []
    total_batches = (total_items + batch_size - 1) // batch_size
    for i in range(total_batches):
        batch = target_data[i*batch_size : (i+1)*batch_size]
        prompts = [build_prompt(item['instruction'], item['input']) for item in batch]
        instructions = [item['instruction'] for item in batch]
        inputs = [item['input'] for item in batch]
        expecteds = [item['output'] for item in batch]
        batch_res = evaluate_batch_logic(prompts, instructions, inputs, expecteds, vocab, infer_fn)
        all_records.extend(batch_res)
        if (i + 1) % 5 == 0: print(f"Processed batch {i+1}/{total_batches}")
    
    # Only calculate averages for the retained metrics
    metric_keys = ["ROUGE-L", "BLEU", "F1"]
    avg_metrics = {k: [] for k in metric_keys}
    for rec in all_records:
        for k in metric_keys: avg_metrics[k].append(rec["metrics"][k])
    final_avgs = {k: np.mean(v) if v else 0.0 for k, v in avg_metrics.items()}
    save_results(model_name, all_records, final_avgs, time.time() - start_time)
    print(f"Completed {model_name}")
    for k, v in final_avgs.items(): print(f"  {k}: {v:.4f}")
    return final_avgs

def run_llama_lora(data, vocab):
    print("\n--- Starting Llama-2-7B + LoRA Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    print("Loading Base Model...")
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None 
    ).to(DEVICE)

    print("Loading LoRA Adapter (PeftModel.from_pretrained)...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.to(DEVICE)
    model.eval()
    model.config.use_cache = False 
    
    def infer(batch_prompts):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.3, top_p=0.95, repetition_penalty=1.2
            )
        res = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            res.append(decoded)
        return res

    process_dataset(data, vocab, infer, "Llama-2-7b-LoRA")
    
    del model
    del base_model
    del tokenizer
    aggressive_cleanup()

def run_llama_base(data, vocab):
    print("\n--- Starting Llama-2-7B-Base Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None 
    ).to(DEVICE)
    model.eval()
    
    def infer(batch_prompts):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.3, top_p=0.95
            )
        res = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            res.append(decoded)
        return res

    process_dataset(data, vocab, infer, "Llama-2-7b-Base")
    del model
    del tokenizer
    aggressive_cleanup()

def run_galactica(data, vocab):
    print("\n--- Starting Galactica-6.7B Evaluation ---")
    aggressive_cleanup()
    
    tokenizer = AutoTokenizer.from_pretrained(GALACTICA_PATH)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        GALACTICA_PATH,
        torch_dtype=torch.float16,
        device_map=None
    ).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    def infer(batch_prompts):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CUTOFF_LEN).to(DEVICE)
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
        res = []
        for j, out in enumerate(outputs):
            input_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            res.append(decoded)
        return res

    process_dataset(data, vocab, infer, "Galactica-6.7b")
    del model, tokenizer
    aggressive_cleanup()

def run_gpt(data, vocab):
    print("\n--- Starting GPT-3.5 Evaluation ---")
    def infer(batch_prompts):
        responses = []
        for prompt in batch_prompts:
            payload = {
                "model": GPT_MODEL_NAME,
                "messages": [{"role": "system", "content": "You are a helpful and precise protein biology assistant."}, {"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
            try:
                time.sleep(1) 
                r = requests.post(GPT_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
                data = r.json()
                if "choices" in data and data["choices"]: responses.append(data["choices"][0]["message"]["content"].strip())
                else: responses.append("")
            except Exception as e:
                print(f"GPT Error: {e}")
                responses.append("")
        return responses
    process_dataset(data, vocab, infer, "GPT-3.5", batch_size=5)


def main():
    print(f"Using Logical Device: {DEVICE}") 
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    print("Building Keyword Vocabulary from Ground Truth...")
    vocab = build_vocab(full_data)
    print(f"Vocabulary Size: {len(vocab)}")
    
    run_llama_base(full_data, vocab)
    run_llama_lora(full_data, vocab)
    run_galactica(full_data, vocab)
    run_gpt(full_data, vocab)
    
    print("\nAll evaluations completed successfully.")

if __name__ == "__main__":
    main()