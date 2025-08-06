# ===================================================================================
#
#          Med-LLM 互動式聊天測試腳本 (優化版)
#
#  此版本使用 `transformers.pipeline` 進行推論，程式碼更簡潔、更穩健，
#  並提供了一個可以連續對話的互動式介面。
#
# ===================================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

# ===================================================================================
# 1. 設定模型路徑 (與訓練腳本一致)
# ===================================================================================
# 基礎模型路徑
base_model_id = "./Llama-3.1-8B-Instruct"
# 最終的 DPO Adapter 路徑
adapter_path = "llama-3.1-8b-med-robot-adapter-dpo-v2/final"

# ===================================================================================
# 2. 準備量化設定與載入模型
# ===================================================================================
# 使用與訓練時完全一致的 BNB 量化設定，以確保結果一致性
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"--- 正在從 {base_model_id} 載入基礎模型與 Tokenizer ---")
# 載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"  # 自動將模型分配到可用的 GPU
)
# 載入分詞器
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 關鍵一步：將 LoRA Adapter 應用到基礎模型上
print(f"--- 正在從 {adapter_path} 應用 DPO LoRA Adapter ---")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("--- 您的專屬 Med-LLM 已準備就緒！ ---")

# ===================================================================================
# 3. 使用 `pipeline` 建立一個方便的文字生成器
# ===================================================================================
# pipeline 是 transformers 中進行推論的推薦方式，它會自動處理大部分細節
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # 設定生成回覆的最大長度
    do_sample=True,  # 啟用採樣，讓回答更自然
    temperature=0.6,  # 溫度稍低，讓醫療建議更穩定
    top_p=0.9,  # Top-p 採樣
)

# ===================================================================================
# 4. 建立互動式聊天循環
# ===================================================================================
# 設定固定的 system prompt
system_prompt = "你是一個專業、有同理心且謹慎的送藥機器人。你的回答應該要清晰、有條理，並且總是優先考慮患者的安全。在提供任何建議時，都要提醒使用者諮詢專業醫師。"
messages = [{"role": "system", "content": system_prompt}]

print("\n" + "=" * 50)
print("     歡迎使用 Med-LLM 互動式聊天！")
print("  (輸入 'exit' 或 'quit' 來結束對話)")
print("=" * 50)

while True:
    # 接收使用者輸入
    user_input = input("\n您 (User): ")
    if user_input.lower() in ["exit", "quit"]:
        print("\nMed-LLM: 很高興為您服務，再見！")
        break

    # 將使用者輸入加入對話歷史
    messages.append({"role": "user", "content": user_input})

    # 使用 pipeline 進行推論
    # pipeline 會自動應用對話模板
    print("\nMed-LLM (正在思考...):")
    result = pipe(messages)

    # 輸出模型的回答
    # result[0]['generated_text'] 是一個包含完整對話的列表
    # 我們只需要最後一條 assistant 的回覆
    assistant_response = result[0]['generated_text'][-1]['content']
    print(assistant_response)

    # 將模型的回答也加入對話歷史，這樣才能進行多輪對話
    messages.append({"role": "assistant", "content": assistant_response})