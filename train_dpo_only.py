# ===================================================================================
#
#          Med-LLM 獨立 DPO 訓練腳本 (修正版)
#
#  此腳本假設您已經成功執行過 SFT 階段，並擁有了 SFT Adapter。
#  它專門用於進行 DPO 階段的訓練與微調。
#
# ===================================================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
import os

# ===================================================================================
# 1. 參數設定 (請確保路徑正確)
# ===================================================================================
# 基礎模型路徑
BASE_MODEL_ID = "./Llama-3.1-8B-Instruct"

# 【關鍵】SFT 階段產生的 Adapter 路徑
SFT_ADAPTER_PATH = "llama-3.1-8b-med-robot-adapter-sft/final"

# DPO 階段要使用的資料集
DPO_DATASET_PATH = "dpo_dataset.json"

# DPO 訓練完成後，新 Adapter 的儲存名稱
NEW_DPO_ADAPTER_NAME = "llama-3.1-8b-med-robot-adapter-dpo-v2" # 建議用新名字，避免覆蓋

# ===================================================================================
# 2. 準備量化與 LoRA 設定
# ===================================================================================
# 量化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA 設定 (必須與 SFT 階段完全一致！)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ===================================================================================
# 3. 載入 Tokenizer 和處理資料集
# ===================================================================================
print("--- 載入 Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO 訓練 padding 在左邊

def create_dpo_conversations(row):
    system_message = "你是一個專業的送藥機器人。" # System prompt 應與您期望的一致
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["prompt"]}
    ]
    row["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    row["chosen"] = tokenizer.apply_chat_template(prompt_messages + [{"role": "assistant", "content": row["chosen"]}], tokenize=False, add_generation_prompt=False)
    row["rejected"] = tokenizer.apply_chat_template(prompt_messages + [{"role": "assistant", "content": row["rejected"]}], tokenize=False, add_generation_prompt=False)
    return row

print(f"--- 正在載入並預處理 DPO 資料集: {DPO_DATASET_PATH} ---")
dpo_dataset = load_dataset('json', data_files=DPO_DATASET_PATH, split='train')
dpo_dataset = dpo_dataset.map(create_dpo_conversations)
print("--- DPO 資料集準備完成 ---")

# ===================================================================================
# 4. 載入用於 DPO 訓練的模型
# ===================================================================================
print("--- 載入 Policy Model (Base Model + SFT Adapter) ---")
# 1. 先載入 4-bit 量化的基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
# 2. 將 SFT Adapter 套用上去，得到我們要繼續訓練的 policy model
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
print("--- Policy Model 載入完成 ---")

# ===================================================================================
# 5. 設定 DPO 訓練參數並初始化 DPOTrainer
# ===================================================================================
# DPO 訓練參數
dpo_training_args = DPOConfig(
    output_dir=NEW_DPO_ADAPTER_NAME,
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
    padding_value=tokenizer.pad_token_id, # 明確指定 padding value
)

# 初始化 DPOTrainer (採用我們最終成功的配置)
dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=None, # 設為 None，讓 trl 自動處理
    args=dpo_training_args,
    train_dataset=dpo_dataset,
    peft_config=lora_config, # 提供 LoRA 配置，讓 trl 知道如何處理 adapter

)

# ===================================================================================
# 6. 開始訓練並儲存
# ===================================================================================
print("--- DPO 訓練即將開始！ ---")
dpo_trainer.train()
print("--- DPO 訓練完成！ ---")

final_path = os.path.join(NEW_DPO_ADAPTER_NAME, "final")
dpo_trainer.save_model(final_path)

print("\n" + "*"*80)
print(f"✅ 恭喜！獨立 DPO 訓練完成！")
print(f"   - 基於 SFT Adapter: {SFT_ADAPTER_PATH}")
print(f"   - 新的 DPO Adapter 已儲存至: {final_path}")
print("*"*80)