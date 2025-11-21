# Med-LLM: 基於大型語言模型之智慧醫療送藥機器人系統

## 摘要

本研究提出一個專門針對醫療領域設計的大型語言模型系統 Med-LLM，旨在提供智慧化的藥物遞送與諮詢服務。本系統基於 Meta 公司開發的 Llama 3.1-8B-Instruct 基礎模型，採用參數高效微調技術（Parameter-Efficient Fine-Tuning, PEFT）中的低秩適應（Low-Rank Adaptation, LoRA）方法，結合監督式微調（Supervised Fine-Tuning, SFT）與直接偏好優化（Direct Preference Optimization, DPO）兩階段訓練策略，建構出一個具備醫療專業知識、安全可靠且富有同理心的對話式人工智慧系統。

## 1. 研究背景與動機

### 1.1 研究背景

隨著人口老化與慢性病患者數量持續增加，醫療機構面臨日益沉重的人力負擔。藥物管理與遞送服務是醫療照護中的重要環節，然而傳統的人工作業模式不僅效率有限，且容易因人為疏失導致用藥安全問題。近年來，人工智慧技術的快速發展為醫療領域帶來新的解決方案，特別是大型語言模型（Large Language Models, LLMs）在自然語言理解與生成方面展現出卓越的能力，為建構智慧醫療對話系統提供了技術基礎。

### 1.2 研究動機

本研究旨在解決以下關鍵問題：

1. **醫療專業知識整合**：如何將複雜的醫療專業知識有效整合至大型語言模型中。
2. **患者安全保障**：如何確保系統提供的資訊符合醫療安全標準，避免產生有害建議。
3. **人性化互動**：如何使系統具備同理心，提供溫暖且專業的醫療服務體驗。
4. **計算資源優化**：如何在有限的計算資源下，實現高效的模型訓練與部署。

## 2. 系統架構

### 2.1 基礎模型

本系統採用 Meta 公司開發的 Llama 3.1-8B-Instruct 作為基礎模型。Llama 3.1 系列是基於 Transformer 架構的自回歸語言模型，具有 80 億參數規模，支援多語言理解與生成，並已針對指令遵循任務進行預訓練優化。

### 2.2 參數高效微調技術

考量到全參數微調（Full Fine-Tuning）所需的龐大計算資源與記憶體需求，本研究採用 LoRA（Low-Rank Adaptation）技術進行參數高效微調。LoRA 透過在模型的注意力層中插入低秩矩陣，僅訓練少量新增參數，即可達到接近全參數微調的效果，同時大幅降低記憶體使用量與訓練時間。

**LoRA 配置參數：**

- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- 目標模組：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 2.3 量化技術

為進一步優化記憶體使用，本系統採用 4-bit 量化技術，使用 BitsAndBytes 函式庫實現 NF4（Normal Float 4-bit）量化方法，配合 bfloat16 計算精度，在保持模型效能的同時，將記憶體需求降低約 75%。

**量化配置：**

- 量化位元數：4-bit
- 量化類型：NF4
- 計算精度：bfloat16
- 雙重量化：啟用

## 3. 訓練方法論

### 3.1 兩階段訓練策略

本系統採用兩階段訓練策略，結合監督式微調與直接偏好優化，以達到最佳的模型效能與安全性。

#### 3.1.1 第一階段：監督式微調（SFT）

監督式微調階段的目標是使基礎模型學習醫療領域的專業知識與對話模式。本階段使用精心設計的醫療對話資料集，涵蓋多種臨床場景與藥物類型。

**訓練參數：**

- 學習率：2e-4
- 訓練輪數：2
- 批次大小：1
- 梯度累積步數：4
- 學習率調度器：Cosine
- 優化器：Paged AdamW 8-bit
- 預熱比例：0.05
- 混合精度訓練：FP16

#### 3.1.2 第二階段：直接偏好優化（DPO）

直接偏好優化階段旨在提升模型回應的安全性與專業性。DPO 是一種無需獎勵模型的強化學習方法，透過對比優選（chosen）與劣選（rejected）回應對，直接優化模型的輸出分佈，使其更符合人類偏好。

**訓練參數：**

- 學習率：5e-6
- 訓練輪數：1
- 批次大小：1
- 梯度累積步數：2
- Beta 參數：0.1
- 學習率調度器：Cosine
- 優化器：Paged AdamW 8-bit
- 預熱比例：0.1
- 混合精度訓練：FP16
- Padding 方向：左側

### 3.2 資料集

#### 3.2.1 監督式微調資料集

監督式微調資料集包含 830 筆高品質醫療對話樣本，涵蓋以下場景：

- 藥局服務（25%）
- 病房照護（20%）
- 急診處理（20%）
- 居家護理（15%）
- 兒童專科（10%）
- 孕婦護理（10%）

每筆資料包含以下欄位：

- **instruction**：任務指令，描述送藥機器人當前的服務場景
- **input**：輸入資訊，如藥品名稱與劑量
- **output**：期望的模型回應，包含用藥指導與注意事項
- **system**：系統提示詞，定義機器人的角色與行為規範

**資料範例：**

```json
{
  "instruction": "我正在為您送達阿司匹林，請說明用藥注意事項。",
  "input": "藥品：阿司匹林 100mg",
  "output": "您好！我為您送來阿司匹林 100mg。這是用於血栓預防的藥物，主要注意事項：1)可能的副作用包括胃刺激；2)重要提醒：飯後服用；3)如有任何不適請立即聯繫醫師；4)請按時服用，不要擅自調整劑量。祝您早日康復！",
  "system": "你是專業的送藥機器人，正在藥局為老年人患者提供藥物遞送服務。"
}
```

#### 3.2.2 直接偏好優化資料集

DPO 資料集包含多組對比樣本，每組包含一個提示（prompt）、一個優選回應（chosen）與一個劣選回應（rejected）。優選回應強調安全性、專業性與同理心，而劣選回應則可能存在資訊不完整、偏離重點或安全性考量不足等問題。

**資料範例：**

```json
{
  "prompt": "我中午的降血壓藥忘記吃了，怎麼辦？",
  "chosen": "別擔心，忘記吃藥是常有的事。最重要的原則是『千萬不要一次吃兩倍的劑量』。如果現在離您下次服藥的時間比較近了，最安全的做法是直接跳過這次，等正常時間再吃下次的藥就好。",
  "rejected": "別著急！你可以查一下你的藥物清單，看看今天的藥物配送時間和你的藥物名。然後，我會幫你送來你的藥物。"
}
```

## 4. 實驗環境與系統需求

### 4.1 硬體需求

**最低配置：**

- GPU：12GB VRAM（如 NVIDIA RTX 3060 12GB）
- 系統記憶體：16GB RAM
- 儲存空間：100GB 可用空間
- CUDA 版本：11.8 或更高

**建議配置：**

- GPU：24GB VRAM（如 NVIDIA RTX 4090）
- 系統記憶體：32GB RAM 或更高
- 儲存空間：200GB SSD
- CUDA 版本：12.0 或更高

### 4.2 軟體環境

本系統基於 Python 3.8 或更高版本開發，主要依賴以下函式庫：

- PyTorch >= 2.0.0：深度學習框架
- Transformers >= 4.35.0：Hugging Face 模型與分詞器
- Datasets >= 2.14.0：資料集處理工具
- Accelerate >= 0.21.0：分散式訓練加速
- PEFT >= 0.6.0：參數高效微調工具
- TRL >= 0.7.0：強化學習訓練工具
- BitsAndBytes >= 0.41.0：量化優化工具
- SentencePiece >= 0.1.99：文本分詞工具
- Protobuf >= 3.20.0：資料序列化工具

## 5. 系統部署與使用

### 5.1 環境安裝

#### 步驟 1：安裝 PyTorch 與 CUDA 支援

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 步驟 2：安裝相關依賴

```bash
pip install -r requirements.txt
```

#### 步驟 3：驗證 CUDA 環境

```python
import torch
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"GPU 數量：{torch.cuda.device_count()}")
print(f"GPU 名稱：{torch.cuda.get_device_name(0)}")
```

### 5.2 模型準備

本系統需要下載 Llama 3.1-8B-Instruct 基礎模型。可透過以下方式取得：

#### 方法 1：使用 Hugging Face CLI

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruct
```

#### 方法 2：使用 Git LFS

```bash
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ./Llama-3.1-8B-Instruct
```

#### 方法 3：程式化下載

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

### 5.3 模型訓練

#### 5.3.1 整合式訓練（建議）

執行整合式訓練腳本，一次完成 SFT 與 DPO 兩階段訓練：

```bash
python combine.py
```

此腳本將自動執行以下流程：

1. 載入基礎模型與 LoRA 配置
2. 執行監督式微調階段
3. 儲存 SFT Adapter
4. 釋放記憶體
5. 執行直接偏好優化階段
6. 儲存最終 DPO Adapter

**預估訓練時間：**

- SFT 階段：2-4 小時（依硬體配置而定）
- DPO 階段：1-2 小時
- 總計：約 4-6 小時

#### 5.3.2 分階段訓練

若需要更靈活的訓練控制，可分別執行各階段訓練：

**第一階段：監督式微調**

```bash
python train_med_llm.py
```

**第二階段：直接偏好優化**

```bash
python train_dpo_only.py
```

### 5.4 模型測試與評估

使用互動式測試腳本進行模型效能評估：

```bash
python test_model.py
```

此腳本提供以下功能：

- 多輪連續對話
- 上下文記憶
- 即時回應生成
- 互動式命令介面（輸入 'exit' 或 'quit' 結束對話）

## 6. 專案結構

```
med_gpt/
├── train_med_llm.py              # SFT 階段訓練腳本
├── train_dpo.py                  # DPO 階段訓練腳本（含參考模型）
├── train_dpo_only.py             # DPO 階段獨立訓練腳本
├── combine.py                    # 整合式兩階段訓練腳本
├── test_model.py                 # 互動式測試腳本
├── med_dataset.json              # 監督式微調資料集
├── dpo_dataset.json              # 直接偏好優化資料集
├── requirements.txt              # Python 依賴套件清單
└── README.md                     # 專案說明文件
```

## 7. 訓練輸出

訓練完成後，系統將產生以下輸出：

### 7.1 SFT 階段輸出

```
llama-3.1-8b-med-robot-adapter-sft/
├── final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
└── checkpoint-*/
```

### 7.2 DPO 階段輸出

```
llama-3.1-8b-med-robot-adapter-dpo/
├── final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
└── checkpoint-*/
```

**注意事項：** 最終推理時應使用 DPO 階段產生的 Adapter，因其已經過安全性與專業性優化。

## 8. 技術特點與創新

### 8.1 記憶體效率

透過結合 LoRA 與 4-bit 量化技術，本系統僅需 12GB VRAM 即可完成訓練，相較於全參數微調節省約 85% 的記憶體使用量。

### 8.2 訓練效率

採用梯度累積與混合精度訓練技術，在有限批次大小下仍可維持訓練穩定性與收斂速度。

### 8.3 安全性保障

透過 DPO 階段的偏好學習，系統能夠自動識別並避免產生不安全或不適當的醫療建議，始終優先考慮患者安全。

### 8.4 專業性與同理心

系統訓練過程強調醫療專業知識的準確性，同時注重語言表達的同理心與溫暖度，提供人性化的互動體驗。

## 9. 限制與未來工作

### 9.1 當前限制

1. **非診斷工具**：本系統僅供參考，不能替代專業醫療診斷與治療。
2. **語言限制**：目前主要支援繁體中文，多語言支援有待擴展。
3. **知識更新**：模型知識截止於訓練資料收集時間，無法即時更新最新醫療資訊。
4. **計算資源**：儘管已優化記憶體使用，仍需具備 GPU 的計算環境。

### 9.2 未來研究方向

1. **多模態整合**：整合視覺與語音資訊，提供更全面的醫療服務。
2. **知識庫擴展**：建立動態知識更新機制，確保醫療資訊的時效性。
3. **個人化服務**：根據患者歷史記錄提供個人化的用藥建議。
4. **跨語言支援**：擴展至多語言環境，服務更廣泛的使用者群體。
5. **實體機器人整合**：與自主導航送藥機器人結合，實現端到端的智慧醫療服務。

## 10. 安全聲明與責任限制

### 10.1 醫療免責聲明

本系統僅供研究與輔助參考用途，不構成任何形式的醫療建議、診斷或治療建議。使用者在任何情況下均應：

1. 諮詢合格的醫療專業人員
2. 遵循醫師處方與指示
3. 不得將系統輸出作為醫療決策的唯一依據
4. 緊急情況請立即就醫或撥打急救電話

### 10.2 使用限制

本系統嚴禁用於：

1. 替代專業醫療診斷與治療
2. 處方藥物或調整藥物劑量
3. 處理危及生命的緊急情況
4. 任何可能危害患者安全的場景

### 10.3 法律責任

使用者需自行承擔使用本系統所產生的一切風險與責任。開發團隊不對因使用本系統而導致的任何直接或間接損失負責。

## 11. 開源授權

本專案採用 MIT License 授權條款。

```
MIT License

Copyright (c) 2024 Med-LLM Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 12. 致謝

本研究感謝以下機構與專案的技術支援：

- **Meta AI**：提供 Llama 3.1 基礎模型
- **Hugging Face**：提供 Transformers 函式庫與模型託管服務
- **Microsoft**：提供 DeepSpeed 與 Accelerate 訓練優化工具
- **NVIDIA**：提供 CUDA 計算平台
- **開源社群**：PyTorch、PEFT、TRL、BitsAndBytes 等專案的貢獻者

感謝所有參與資料標註與驗證的醫療專業人員，以及為本專案提供寶貴建議的研究人員與開發者。

## 13. 聯絡資訊

如有技術問題或合作意向，歡迎透過 GitHub Issues 提出討論。

## 14. 參考文獻

1. Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288.
2. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv preprint arXiv:2106.09685.
3. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv preprint arXiv:2305.18290.
4. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv preprint arXiv:2305.14314.

---

最後更新日期：2024 年
