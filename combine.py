# ===================================================================================
#
#          Med-LLM 整合式訓練腳本 (最終保證版)
#
#  此版本基於所有錯誤日誌的最終分析，採用 trl 函式庫對 PEFT 最標準的設計，
#  將 ref_model 的創建與管理完全交由 DPOTrainer 自動處理，以解決底層衝突。
#
# ===================================================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, DPOTrainer, DPOConfig
import os

# ===================================================================================
# 1. 全局參數設定
# ===================================================================================
BASE_MODEL_ID = "./Llama-3.1-8B-Instruct"
SFT_DATASET_PATH = "med_dataset.json"
SFT_ADAPTER_NAME = "llama-3.1-8b-med-robot-adapter-sft"
DPO_DATASET_PATH = "dPO_dataset.json"
DPO_ADAPTER_NAME = "llama-3.1-8b-med-robot-adapter-dpo"

# 量化設定，為 VRAM 保駕護航
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ===================================================================================
# 2. 載入共用的 Tokenizer
# ===================================================================================
print("--- 載入共用的 Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================================================================================
#
#                     第一階段：SFT (監督式微調)
#
# ===================================================================================
print("\n" + "="*80)
print("                      啟動第一階段：SFT 微調")
print("="*80 + "\n")

# VRAM 最佳化：降低 LoRA Rank
sft_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer.padding_side = "right"
print(f"--- SFT 階段 Tokenizer padding side 設定為: {tokenizer.padding_side} ---")

print(f"--- 載入基礎模型用於 SFT: {BASE_MODEL_ID} ---")
sft_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
print("--- SFT 基礎模型載入完成 ---")

sft_training_args = TrainingArguments(
    output_dir=SFT_ADAPTER_NAME,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=20,
    save_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

print(f"--- 正在載入 SFT 資料集: {SFT_DATASET_PATH} ---")
sft_dataset = load_dataset('json', data_files=SFT_DATASET_PATH, split='train')

def sft_formatting_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        system_message = example['system'][i] if 'system' in example and example['system'][i] else '你是一個專業的送藥機器人。'
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{example['instruction'][i]}\n{example.get('input', [''])[i]}"},
            {"role": "assistant", "content": example['output'][i]}
        ]
        output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return output_texts

sft_trainer = SFTTrainer(
    model=sft_base_model,
    args=sft_training_args,
    train_dataset=sft_dataset,
    peft_config=sft_lora_config,
    formatting_func=sft_formatting_func,
)

print("--- SFT 訓練即將開始！ ---")
sft_trainer.train()
print("--- SFT 訓練完成！ ---")

sft_final_path = os.path.join(SFT_ADAPTER_NAME, "final")
sft_trainer.save_model(sft_final_path)
print(f"✅ SFT Adapter 已儲存至: {sft_final_path}")

del sft_base_model, sft_trainer
torch.cuda.empty_cache()
print("--- SFT 模型與訓練器已從記憶體中釋放 ---")


# ===================================================================================
#
#                     第二階段：DPO (直接偏好優化)
#
# ===================================================================================
print("\n" + "="*80)
print("                      啟動第二階段：DPO 優化")
print("="*80 + "\n")

tokenizer.padding_side = "left"
print(f"--- DPO 階段 Tokenizer padding side 已切換為: {tokenizer.padding_side} ---")

def create_dpo_conversations(row):
    system_message = "你是一個專業的送藥機器人。"
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["prompt"]}
    ]
    row["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    row["chosen"] = tokenizer.apply_chat_template(prompt_messages + [{"role": "assistant", "content": row["chosen"]}], tokenize=False, add_generation_prompt=False)
    row["rejected"] = tokenizer.apply_chat_template(prompt_messages + [{"role": "assistant", "content": row["rejected"]}], tokenize=False, add_generation_prompt=False)
    return row

print(f"--- 正在載入並預處理 DPO 資料集: {DPO_DATASET_PATH} ---")
raw_dpo_dataset = load_dataset('json', data_files=DPO_DATASET_PATH, split='train')
dpo_dataset = raw_dpo_dataset.map(create_dpo_conversations)
print("--- DPO 資料集準備完成 ---")

print("--- 載入 Policy Model (在 GPU)，這是唯一需要手動載入的模型 ---")
dpo_policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
dpo_policy_model = PeftModel.from_pretrained(dpo_policy_model, sft_final_path, is_trainable=True)
print("--- Policy Model 載入完成 ---")


dpo_training_args = DPOConfig(
    output_dir=DPO_ADAPTER_NAME,
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    padding_value=tokenizer.pad_token_id, # 解決 padding_value 缺失問題
)

# ★★★★★★★★★★★★★★★★★★★★★★★★★ 最終的初始化方式 ★★★★★★★★★★★★★★★★★★★★★★★★
# 我們將 ref_model 設為 None，並提供 peft_config，完全遵循 trl 的標準 PEFT 訓練流程。
# DPOTrainer 會在內部自動創建一個非訓練狀態的參考模型。
dpo_trainer = DPOTrainer(
    model=dpo_policy_model,
    ref_model=None,
    args=dpo_training_args,
    train_dataset=dpo_dataset,
    peft_config=sft_lora_config,
)

print("--- DPO 訓練即將開始！ ---")
dpo_trainer.train()
print("--- DPO 訓練完成！ ---")

dpo_final_path = os.path.join(DPO_ADAPTER_NAME, "final")
dpo_trainer.save_model(dpo_final_path)

print("\n" + "*"*80)
print(f"✅ 恭喜！整合式訓練全部完成！")
print(f"   - SFT Adapter 位於: {sft_final_path}")
print(f"   - 最終 DPO Adapter 位於: {dpo_final_path}")
print("   - 您在後續推理時，應該使用最終的 DPO Adapter。")
print("*"*80)