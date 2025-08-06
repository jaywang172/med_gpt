import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os

# ===================================================================================
# 1. 参数设定
# ===================================================================================
MODEL_ID = "./Llama-3.1-8B-Instruct"
DATASET_PATH = "med_dataset.json"
NEW_MODEL_NAME = "llama-3.1-8b-med-robot-adapter"

# ===================================================================================
# 2. 模型载入设定
# ===================================================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ===================================================================================
# 3. 载入模型与分词器
# ===================================================================================
print(f"--- 正在从本地路徑載入基礎模型: {MODEL_ID} ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
# 注意：对于 SFTTrainer 的自动 packing，padding_side='right' 是推荐设置
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = get_peft_model(model, lora_config)
print("--- 模型與 LoRA 配置載入完成 ---")

# ===================================================================================
# 4. 载入数据集 (注意：这次我们不做任何 map 操作！)
# ===================================================================================
print(f"--- 正在載入原始資料集: {DATASET_PATH} ---")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
print("--- 原始資料集準備完成 ---")

# ===================================================================================
# 5. 设定训练参数
# ===================================================================================
training_args = TrainingArguments(
    output_dir=NEW_MODEL_NAME,
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


# ===================================================================================
# 6. 初始化 SFTTrainer (最终的、权威的、最精简的用法)
# ===================================================================================

# 这个函数是关键，它告诉 SFTTrainer 如何将我们 JSON 中的一笔资料转换成 Llama 3 的对话格式
# SFTTrainer 会在内部自动调用这个函数
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        system_message = example['system'][i] if 'system' in example and example['system'][i] else '你是一個專業的送藥機器人。'

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{example['instruction'][i]}\n{example.get('input', [''])[i]}"},
            {"role": "assistant", "content": example['output'][i]}
        ]
        # 注意：这里我们只返回文本，tokenize 的工作完全交给 SFTTrainer
        output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return output_texts


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # <-- 传入最原始的数据集
    peft_config=lora_config,
    # 关键参数：
    formatting_func=formatting_prompts_func,  # <-- 告诉它如何将原始数据转换成对话字符串 # <-- 在这里指定最大长度
)

print("--- 一切準備就緒，即將開始訓練您的專屬 Med-LLM！ ---")
trainer.train()
print("--- 訓練完成！ ---")

# ===================================================================================
# 7. 储存最终的 LoRA Adapter
# ===================================================================================
final_path = os.path.join(NEW_MODEL_NAME, "final")
trainer.save_model(final_path)
print(f"✅ 恭喜！您專屬的 Med-LLM (LoRA Adapter) 已儲存至: {final_path}")