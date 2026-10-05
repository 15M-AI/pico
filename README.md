# Pico 15M AI Assistant 🤖

**Pico** is an open-source, lightweight 15M parameter Causal Transformer language model built from scratch using PyTorch. Designed for fast CPU execution and local deployment, Pico has been fine-tuned on general-purpose instruction datasets (`ultrachat_200k`) to deliver interactive terminal chat experiences.

---

## 📐 Architecture Specifications

- **Parameters:** ~15,000,000
- **Layers (`n_layer`):** 6 Transformer Blocks
- **Attention Heads (`n_head`):** 6
- **Embedding Dimension (`n_embd`):** 384
- **Context Window (`block`):** 256 tokens
- **Vocabulary Size:** 8,192 tokens (Custom BPE Tokenizer)
- **Token Format:** `<user>{input}<pico>` turn structure with `<bos>` and `<eos>` special tokens.

---

## 📊 Training Details

- **Dataset:** 30,000 dialogue pairs from `HuggingFaceH4/ultrachat_200k`
- **Optimizer:** AdamW (`lr=5e-4`, `weight_decay=0.01`)
- **Duration:** 5 Epochs in Google Colab (T4 GPU)
- **Loss trajectory:**
  - **Epoch 1:** Loss 5.2438
  - **Epoch 2:** Loss 4.0153
  - **Epoch 3:** Loss 3.5179
  - **Epoch 4:** Loss 3.1616
  - **Epoch 5:** **Loss 2.8707**

---

## 🚀 Local Setup & Installation

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone git@github.com:15M-AI/Pico.git
cd Pico

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

```

### 2. Download Model Weights & Tokenizer

Download `pico-15m.pt` and `pico_tokenizer.json` directly from Hugging Face:

```bash
pip install huggingface_hub
hf download 15MAI/Pico pico-15m.pt --local-dir .
hf download 15MAI/Pico pico_tokenizer.json --local-dir .

```

### 3. Install Dependencies

```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install tokenizers numpy

```

---

## 💬 Running Interactive Chat

Launch the interactive terminal interface:

```bash
python app.py

```

**Example Conversation:**

```text
--- Pico 15M General Assistant (15M-AI) ---
Escribe 'exit' para salir.

You: Hello pico
Pico: I am ready to help you write and structure your context.

You: How can you help me?
Pico: I can assist with answering general questions and providing structured text responses.

```

---

## 📁 Repository Structure

```text
├── app.py                # Main CLI inference entrypoint
├── pico-15m.pt           # Model weights checkpoint (PyTorch)
├── pico_tokenizer.json   # BPE Tokenizer configuration
├── .gitignore            # Git exclusion definitions
└── README.md             # Project documentation

```

---

## 🔗 Model Weights & Links

* **Hugging Face Hub:** [15MAI/Pico](https://huggingface.co/15MAI/Pico)
* **GitHub Repository:** [15M-AI/Pico](https://www.google.com/search?q=https://github.com/15M-AI/Pico)

---

## 📄 License

This project is released under the Apache-2.0 License.

