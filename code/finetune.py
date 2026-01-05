import os
import torch
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

print(f"Transformers version: {transformers.__version__}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


os.environ["WANDB_DISABLED"] = "true"

model_path = "/workspace/contrast/Llama-2-7b-chat-hf"
output_path = "./merge-model-pro-new-0"


is_bf16_supported = torch.cuda.is_bf16_supported()
print(f"BF16: {is_bf16_supported}")

print("正在加载模型...")
model = transformers.LlamaForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16 if is_bf16_supported else torch.float16,
    attn_implementation="flash_attention_2" 
)


model.enable_input_require_grads()
model.config.use_cache = False 

tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


if not is_bf16_supported:
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

model.print_trainable_parameters()

def tokenize_function(examples):

    input_text = examples.get('input', "")
    if input_text is None:
        input_text = ""

    prompt = f"### Instruction:\n{examples['instruction']}\n### Input:\n{input_text}\n### Output:\n"

    full_text = prompt + examples["output"] + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=256,  
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    prompt_ids = tokenizer(prompt, truncation=True, max_length=256, add_special_tokens=True)["input_ids"]
    output_start = len(prompt_ids)

    if output_start < len(labels):
        labels[:output_start] = -100

    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

print("正在处理数据...")
raw_datasets = load_dataset("json", data_files="PP.json")["train"]
raw_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

train_data = raw_datasets["train"].map(
    tokenize_function, 
    remove_columns=["instruction", "input", "output"],
    num_proc=16 
)
eval_data = raw_datasets["test"].map(
    tokenize_function, 
    remove_columns=["instruction", "input", "output"]
)

training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=32, 
    gradient_accumulation_steps=4,
    num_train_epochs=7,
    learning_rate=2e-4,
    
    bf16=is_bf16_supported,
    fp16=not is_bf16_supported,
    max_grad_norm=0.3,
    
    
    report_to=[], 
    logging_steps=500,

    dataloader_num_workers=8,      
    dataloader_pin_memory=True,

    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    save_on_each_node=True,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("开始训练..")
trainer.train()

print(f"正在保存模型至 {output_path}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("保存完成。")
