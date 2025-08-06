Med-LLM: A Specialized Medical Chatbot
![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)

![alt text](https://img.shields.io/badge/🤗%20Transformers-blue.svg)

![alt text](https://img.shields.io/badge/🤗%20PEFT-green.svg)

![alt text](https://img.shields.io/badge/🤗%20TRL-yellow.svg)

This repository contains the code and data to build Med-LLM, a specialized language model designed to act as a professional, empathetic, and cautious medication delivery robot assistant. The model is fine-tuned from Meta-Llama-3.1-8B-Instruct using a two-stage process: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

🚀 Project Overview
The goal of this project is to create a reliable AI assistant for healthcare scenarios. The model is trained to:

Provide Clear Medication Information: Explain drug purposes, side effects, and administration instructions.
Handle Common Patient Queries: Safely address issues like forgotten doses and side effects.
Follow Safety Protocols: Perform patient identity verification and handle emergencies by escalating to professionals.
Communicate with Empathy: Offer support and encouragement to patients with a caring and professional tone.
This is achieved through:

Supervised Fine-Tuning (SFT): The model first learns the core knowledge and conversational style from a diverse dataset of medical instructions and interactions (med_dataset.json).
Direct Preference Optimization (DPO): The model's behavior is then refined for safety and alignment. It learns to prefer safe, helpful responses over incorrect or dangerous ones (dpo_dataset.json).
The entire training process leverages QLoRA for memory efficiency, allowing fine-tuning of an 8B parameter model on consumer-grade hardware.

📂 Project Structure
Generated code
.
├── datasets/
│   ├── dpo_dataset.json      # Dataset for DPO (prompt, chosen, rejected)
│   └── med_dataset.json      # Dataset for SFT (instruction, input, output, system)
│
├── Llama-3.1-8B-Instruct/    # Directory for the base model weights
│
├── llama-3.1-8b-med-robot-adapter-sft/  # Output directory for SFT adapter
├── llama-3.1-8b-med-robot-adapter-dpo-v2/ # Output directory for DPO adapter
│
├── combine.py                # (RECOMMENDED) All-in-one script for SFT and DPO training
├── train_med_llm.py          # Standalone script for SFT training
├── train_dpo_only.py         # Standalone script for DPO training (builds on SFT)
├── test_model.py             # Interactive script to chat with the final model
│
└── README.md                 # This file
Use code with caution.
🛠️ Setup and Installation
1. Prerequisites
Python 3.9 or higher
NVIDIA GPU with CUDA support (at least 16GB VRAM recommended)
Git
2. Clone Repository
Generated bash
git clone <your-repository-url>
cd <your-repository-directory>
Use code with caution.
Bash
3. Set Up Virtual Environment
It is highly recommended to use a virtual environment:

Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Use code with caution.
Bash
4. Install Dependencies
The necessary libraries are listed in the import statements of the scripts. Install them using pip:

Generated bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
Use code with caution.
Bash
5. Download the Base Model
The training scripts expect the base model Meta-Llama-3.1-8B-Instruct to be located in a local directory named ./Llama-3.1-8B-Instruct.

Visit the Meta-Llama-3.1-8B-Instruct Hugging Face page.
Accept the license agreement to gain access to the model.
Download the model weights and place them in the ./Llama-3.1-8B-Instruct directory. You can use git lfs for this:
Generated bash
# Make sure you have git-lfs installed (https://git-lfs.github.com/)
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
Use code with caution.
Bash
⚙️ Training Workflow
The recommended way to train the model is to use the combine.py script, which handles both SFT and DPO stages sequentially.

All-in-One Training (Recommended)
The combine.py script first runs the SFT stage, saves the resulting adapter, clears GPU memory, and then immediately starts the DPO stage using the SFT adapter as a base.

Generated bash
python combine.py
Use code with caution.
Bash
This will produce two adapters:

SFT Adapter: llama-3.1-8b-med-robot-adapter-sft/final/
Final DPO Adapter: llama-3.1-8b-med-robot-adapter-dpo/final/
Standalone Scripts (for debugging or custom workflows)
You can also run each stage separately.

Stage 1: Supervised Fine-Tuning (SFT)
This stage teaches the model the "Med-LLM" persona and knowledge.

Generated bash
python train_med_llm.py
Use code with caution.
Bash
Output: An SFT LoRA adapter saved in llama-3.1-8b-med-robot-adapter/final/.

Stage 2: Direct Preference Optimization (DPO)
This stage refines the SFT model for safety and alignment. It requires the SFT adapter from the previous step. Ensure the SFT_ADAPTER_PATH in train_dpo_only.py is correct.

Generated bash
python train_dpo_only.py
Use code with caution.
Bash
Output: The final DPO LoRA adapter saved in llama-3.1-8b-med-robot-adapter-dpo-v2/final/.

💬 Interactive Testing
Once the DPO training is complete, you can chat with your specialized Med-LLM using the test_model.py script.

Make sure the adapter_path in the script points to your final DPO adapter directory (llama-3.1-8b-med-robot-adapter-dpo-v2/final).

Run the script:

Generated bash
python test_model.py
Use code with caution.
Bash
You will be prompted to enter messages and can interact with the model in a continuous conversation.

Example Interaction:

Generated code
==================================================
     歡迎使用 Med-LLM 互動式聊天！
  (輸入 'exit' 或 'quit' 來結束對話)
==================================================

您 (User): 我中午的降血壓藥忘記吃了，怎麼辦？

Med-LLM (正在思考...):
您好，忘記吃藥是常有的事，請先不要擔心。最重要的原則是「千萬不要一次服用兩倍的劑量」。由於現在已經過了中午，最安全的做法是直接跳過這次忘記的藥，等到下一次正常的服藥時間再服用正確的劑量即可。如果您的醫師有特別的指示，或者您還是很不放心，建議您可以聯絡您的醫師或藥師以獲得最專業的建議。
Use code with caution.
📜 Datasets
med_dataset.json: Used for Supervised Fine-Tuning (SFT). Contains a wide range of scenarios including drug explanations, patient verification, handling side effects, and providing empathetic support.
instruction: The task for the model.
input: Additional context for the instruction.
output: The ideal response (the "label").
system: The system prompt to set the model's persona.
dpo_dataset.json: Used for Direct Preference Optimization (DPO). Contains critical safety-focused scenarios.
prompt: The user's query.
chosen: The preferred, safer, and more helpful response.
rejected: The less desirable or potentially unsafe response.
⚠️ Disclaimer
This project is a proof-of-concept and for research purposes only. The Med-LLM is not a substitute for a qualified medical professional. Do not use its advice for making real-life medical decisions. Always consult with your doctor or pharmacist for any health-related concerns.
