import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import numpy as np
from bert_score import score

INPUT_FILE = "GPT-3.5_details.json" 

def main():
    print(f"Loading data from {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f" Total records: {len(data)}")

    cands = [] 
    refs = []  
    
    print("   Extracting text...")
    for item in data:

        gen = item.get("generated", "")
 
        ref = item.get("output", item.get("expected", ""))

        if not gen.strip(): gen = "." 
        if not ref.strip(): ref = "."
            
        cands.append(gen)
        refs.append(ref)

    print(f"Ready to evaluate {len(cands)} pairs.")
    print("Starting BERTScore calculation (this may take a while)...")


    P, R, F1 = score(cands, refs, lang="en", verbose=True, batch_size=64, device="cuda")

    avg_f1 = F1.mean().item()
    avg_p = P.mean().item()
    avg_r = R.mean().item()

    print("\n" + "="*40)
    print(f"BERTScore Results (N={len(data)})")
    print("="*40)
    print(f"Precision : {avg_p:.4f}")
    print(f"Recall    : {avg_r:.4f}")
    print(f"F1 Score  : {avg_f1:.4f}")
    print("="*40)

    with open("bertscore_result.txt", "w") as f:
        f.write(f"BERTScore F1: {avg_f1:.4f}\n")
        f.write(f"BERTScore Precision: {avg_p:.4f}\n")
        f.write(f"BERTScore Recall: {avg_r:.4f}\n")

if __name__ == "__main__":
    main()
