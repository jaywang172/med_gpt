# --- 建議的 train_dpo.py 修改版 ---

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
import os

# ===================================================================================
# 1. 參數設定 (與您相同)
# ===================================================================================
BASE_MODEL_ID = "./Llama-3.1-8B-Instruct"
SFT_ADAPTER_PATH = "llama-3.1-8b-med-robot-adapter/final"
DPO_DATASET_PATH = "dPO_dataset.json" # 假設此 JSON 格式正確
NEW_DPO_ADAPTER_NAME = "llama-3.1-8b-med-robot-adapter-dpo"

# ===================================================================================
# 2. 載入 Tokenizer (只需要載入一次)
# ===================================================================================
# 載入與基礎模型匹配的 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# DPO 訓練的關鍵設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO 強烈建議 padding 在左邊

# ===================================================================================
# 3. 準備 DPO 資料集 (重要：預處理資料！)
# ===================================================================================
# 這個函數是修正的關鍵：它將結構化的 DPO 資料轉換為 Llama 3 對話格式的字串
def create_dpo_conversations(row):
    system_message = "你是一個專業的送藥機器人。" # 或者從 row 中讀取

    # 將 prompt 轉換為 user message 格式
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["prompt"]}
    ]
    # **非常重要**： apply_chat_template 時，add_generation_prompt=True
    # 這會自動在結尾加上 <|start_header_id|>assistant<|end_header_id|>\n\n
    # 這樣模型才知道要開始生成回答了
    row["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    # 將 chosen 回應加上 assistant 角色並轉換為完整對話
    chosen_messages = prompt_messages + [{"role": "assistant", "content": row["chosen"]}]
    row["chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)

    # 將 rejected 回應加上 assistant 角色並轉換為完整對話
    rejected_messages = prompt_messages + [{"role": "assistant", "content": row["rejected"]}]
    row["rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)

    return row

print(f"--- 正在載入並預處理 DPO 資料集: {DPO_DATASET_PATH} ---")
raw_dpo_dataset = load_dataset('json', data_files=DPO_DATASET_PATH, split='train')
# 使用 .map() 來應用上面的格式轉換函數
dpo_dataset = raw_dpo_dataset.map(create_dpo_conversations)
print("--- DPO 資料集準備完成 ---")


# ===================================================================================
# 4. 載入用於 DPO 訓練的模型 (Policy Model)
# ===================================================================================
print("--- 載入用於 DPO 訓練的 SFT 模型 (Policy Model) ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 這是我們要訓練的模型：Base Model + SFT Adapter
policy_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" # 使用 device_map 分配到 GPU
)
policy_model = PeftModel.from_pretrained(policy_model, SFT_ADAPTER_PATH)
print("--- Policy Model 載入完成 ---")

# ===================================================================================
# 5. 載入參考模型 (Reference Model) - 這是更穩健的做法
# ===================================================================================
print("--- 載入 DPO 參考模型 (Reference Model) ---")
# ref_model 也應該是 Base Model + SFT Adapter，但它在訓練中不更新
# 我們需要為它單獨載入，並確保它在不同的 device_map 上以避免衝突
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    # 注意：如果 GPU 記憶體不足，可以考慮將 ref_model 放在 CPU 或不同的 GPU 上
    # 例如 device_map={"": "cpu"} 或 device_map={"": 1}
    device_map="auto"
)
ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER_PATH)
print("--- Reference Model 載入完成 ---")

# ===================================================================================
# 6. 初始化 DPOTrainer
# ===================================================================================
# 訓練參數可以保持不變
training_args = DPOConfig(
    output_dir=NEW_DPO_ADAPTER_NAME,
    beta=0.1,  # DPO 的關鍵超參數
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=10,
    fp16=True, # 如果您的 GPU 支援，bf16 通常更好
    optim="paged_adamw_8bit",
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model, # <-- 明確傳入參考模型
    args=training_args,
    train_dataset=dpo_dataset, # <-- 傳入處理過的資料集
    # peft_config=None, # 因為 policy_model 已經是 PeftModel，所以這裡不需要
)

# ===================================================================================
# 7. 訓練與儲存 (與您相同)
# ===================================================================================
print("--- 一切準備就緒，即將開始 DPO 訓練！ ---")
trainer.train()
print("--- DPO 訓練完成！ ---")

final_path = os.path.join(NEW_DPO_ADAPTER_NAME, "final")
trainer.save_model(final_path)
print(f"✅ 恭喜！您最終的 Med-LLM (DPO Adapter) 已儲存至: {final_path}")