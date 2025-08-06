# Med-LLM: 專業醫療送藥機器人語言模型

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-yellow.svg)](https://huggingface.co/transformers)

## 📋 專案概述

Med-LLM 是一個基於 Llama 3.1-8B-Instruct 的專業醫療送藥機器人語言模型。本專案採用先進的二階段訓練策略，結合監督式微調 (SFT) 和直接偏好優化 (DPO)，為醫療環境中的送藥機器人提供專業、安全且有同理心的對話能力。

### 🎯 主要特色

- **專業醫療知識**: 針對藥物配送、用藥安全、副作用諮詢等醫療場景優化
- **安全優先**: 強調患者安全，提醒諮詢專業醫師
- **二階段訓練**: SFT + DPO 雙重優化，提升回應品質和安全性
- **記憶體優化**: 支援 4-bit 量化和 LoRA 微調，降低 GPU 記憶體需求
- **多場景適應**: 支援病房、急診、藥局、居家等多種醫療環境

## 🏗️ 專案架構

```
med_gpt/
├── train_med_llm.py        # 第一階段：SFT 監督式微調
├── train_dpo.py           # 第二階段：DPO 直接偏好優化 (完整版)
├── train_dpo_only.py      # 第二階段：DPO 獨立訓練腳本
├── combine.py             # 整合式訓練腳本 (SFT + DPO)
├── test_model.py          # 模型測試與互動式聊天介面
├── med_dataset.json       # SFT 訓練資料集
├── dpo_dataset.json       # DPO 偏好優化資料集
└── README.md              # 專案說明文件
```

## 🚀 快速開始

### 環境需求

```bash
# 核心依賴
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.21.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0

# GPU 記憶體需求
# - 最低: 12GB (4-bit 量化 + LoRA)
# - 推薦: 16GB+ (更穩定的訓練)
```

### 安裝依賴

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft trl bitsandbytes
```

### 模型準備

1. 下載 Llama 3.1-8B-Instruct 模型至 `./Llama-3.1-8B-Instruct/` 目錄

## 🎓 訓練流程

### 方案一：一鍵整合訓練（推薦）

```bash
python combine.py
```

此腳本會自動完成完整的二階段訓練：
1. **SFT 階段**: 使用醫療對話資料進行監督式微調
2. **DPO 階段**: 使用偏好資料進行直接偏好優化

### 方案二：分階段訓練

#### 第一階段：SFT 監督式微調

```bash
python train_med_llm.py
```

- **目的**: 讓模型學習醫療領域的專業知識和對話模式
- **資料**: `med_dataset.json` (830 筆醫療對話資料)
- **輸出**: `llama-3.1-8b-med-robot-adapter/final/`

#### 第二階段：DPO 直接偏好優化

```bash
python train_dpo_only.py
```

- **目的**: 優化模型的回應品質，提升安全性和專業性
- **資料**: `dpo_dataset.json` (偏好對比資料)
- **輸出**: `llama-3.1-8b-med-robot-adapter-dpo-v2/final/`

## 🧪 模型測試

### 互動式測試

```bash
python test_model.py
```

提供友善的命令行互動介面，支援：
- 連續多輪對話
- 即時回應生成
- 專業醫療諮詢模擬

### 測試範例

```
您 (User): 我忘記吃降血壓藥了，現在還能補吃嗎？

Med-LLM: 別擔心，忘記吃藥是很常見的情況。最重要的安全原則是「絕對不要一次吃雙倍劑量」。

如果現在離下次服藥時間還很久（超過一半的間隔時間），通常可以立即補吃。但如果已經快到下次服藥時間了，建議您直接跳過這次，下次正常時間再服用即可。

請記住，任何用藥疑問都建議您諮詢您的主治醫師或藥師，以確保用藥安全。
```

## 📊 資料集說明

### SFT 訓練資料 (`med_dataset.json`)

- **規模**: 830 筆高品質醫療對話
- **場景**: 藥局、病房、急診、居家等多種醫療環境
- **內容**: 藥物配送、用藥指導、副作用說明等

**資料格式**:
```json
{
  "instruction": "我正在為您送達阿司匹林，請說明用藥注意事項。",
  "input": "藥品：阿司匹林 100mg",
  "output": "您好！我為您送來阿司匹林 100mg...",
  "system": "你是專業的送藥機器人，正在藥局為老年人患者提供藥物遞送服務。"
}
```

### DPO 偏好資料 (`dpo_dataset.json`)

- **規模**: 4 筆精心設計的偏好對比範例
- **目的**: 訓練模型選擇更安全、更專業的回應方式

**資料格式**:
```json
{
  "prompt": "我中午的降血壓藥忘記吃了，怎麼辦？",
  "chosen": "別擔心，忘記吃藥是常有的事。最重要的原則是『千萬不要一次吃兩倍的劑量』...",
  "rejected": "別著急！你可以查一下你的藥物清單，看看今天的藥物配送時間..."
}
```

## ⚙️ 技術特色

### 🔧 模型架構
- **基礎模型**: Meta Llama 3.1-8B-Instruct
- **微調方法**: LoRA (Low-Rank Adaptation)
- **量化技術**: 4-bit BitsAndBytesConfig

### 📈 訓練配置
- **LoRA 參數**: r=16, alpha=32, dropout=0.05
- **批次大小**: 1 (梯度累積 2-4 步)
- **學習率**: SFT: 2e-4, DPO: 5e-6
- **優化器**: paged_adamw_8bit

### 🛡️ 安全措施
- **量化優化**: 降低 VRAM 使用量
- **梯度檢查點**: 防止記憶體溢出
- **自動混合精度**: FP16 訓練加速

## 🎯 使用場景

### 醫療機構
- **病房藥物配送**: 為住院患者提供專業用藥指導
- **急診藥物諮詢**: 緊急情況下的快速用藥建議
- **藥局服務**: 門診患者取藥時的專業諮詢

### 居家照護
- **慢性病管理**: 長期用藥患者的日常指導
- **老年人照護**: 針對老年人群的特殊用藥需求
- **用藥提醒**: 智能化的用藥時間和劑量提醒

## ⚠️ 重要聲明

1. **醫療免責**: 本模型僅供參考，不能替代專業醫療建議
2. **安全第一**: 所有用藥問題務必諮詢專業醫師或藥師
3. **持續改進**: 模型會根據臨床回饋持續優化更新
4. **合規使用**: 請確保在符合當地醫療法規的前提下使用

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進專案：

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 文件。

## 🙏 致謝

- Meta AI 的 Llama 3.1 模型
- Hugging Face 的 Transformers 和 TRL 函式庫
- 所有為醫療 AI 發展做出貢獻的開源社群

---

**⚕️ 讓 AI 為醫療服務，讓科技守護健康 ⚕️**
