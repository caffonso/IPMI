# IPMI - LLM specilized on social security legislation



---

## **📘 Model Architecture Overview (Simple Explanation)**

This project fine-tunes a **Qwen 2.5–1.5B** language model and turns it into a *careful, critical, technical assistant*.
The goal is to make the model better at reasoning about:

* **Python code**
* **Linux commands**
* **Chat interactions**
* **Excel formulas**
* **Crypto/market data** (BTC–USDT)
* **General critical thinking** (avoid hallucinations)

Although several training scripts exist, they all follow the **same overall structure**, described below.

---

## **1. Base Model (“The Brain”)**

We start with:

**Qwen2.5-1.5B-Instruct**
— a small, fast, instruction-tuned language model.

It is loaded in **4-bit mode** so it fits on normal GPUs and trains efficiently.

Sometimes we add **LoRA adapters**, which lets us train only small parts of the model instead of the whole thing.

---

## **2. Training Data (“What the Model Learns From”)**

The model learns from **examples** taken from:

* Python tasks
* Linux command tasks
* Chat datasets (GPT-5 style)
* Excel error-correction datasets
* Crypto datasets
* Critical thinking datasets

Some scripts also use **distillation**, where a larger “teacher model” generates synthetic answers.
The smaller Qwen model then learns by imitating those answers.

---

## **3. Unified Format (“How All Examples Look the Same”)**

Even though the data comes from different sources, every example is rewritten into a **simple text format** that includes:

* A short **system instruction** (ex: “You are a critical Python assistant…”)
* A **domain tag** (ex: `[DOMÍNIO: PYTHON]`)
* A **user question or command**
* The **correct answer**, explanation, or reference solution

This makes all training data look similar, which helps the model learn consistent behavior.

Example:

```
Você é uma assistente crítica de Python.
[DOMÍNIO: PYTHON]
Problema: <text>
Solução de referência: <code>
```

---

## **4. Tokenization & Dataset Creation**

Each formatted example is converted into numbers the model can understand (tokens).

We create a dataset where:

* `input_ids` = what the model reads
* `labels` = what the model should learn to predict
* Padding is added so they fit nicely into batches

This is standard **causal language modeling**.

---

## **5. The Training Loop (HuggingFace Trainer)**

All training scripts use the same routine:

1. Load the Qwen model in 4-bit
2. (Optional) Add LoRA adapters
3. Load the dataset
4. Train using HuggingFace’s `Trainer` with:

   * Small batch sizes
   * Gradient accumulation
   * 8-bit optimizer
   * fp16/bf16 support
5. Save the fine-tuned model into `output_dir/`

No matter which script you choose, the training process follows this pattern.

---

## **6. Variants of the Training Pipeline**

Your project includes several training scripts:

### **A. Distillation**

Uses a big teacher model to generate synthetic answers, then trains Qwen on them.

### **B. Multi-Domain (full)**

Trains on Python + Linux + Chat + Crypto (and sometimes Excel).

Good for building a **general critical technical assistant**.

### **C. Multi-Domain Light**

Only Python + Linux.
Faster, simpler, smaller dataset.

Ideal for small GPUs and quick testing.

---

## **7. What You Get at the End**

The output is a **fine-tuned LLM** that:

* Reads code and points out mistakes
* Analyzes Linux commands with security awareness
* Answers like a careful assistant, not a hallucinating one
* Handles technical tasks in multiple domains
* Avoids blindly agreeing with the user
* Follows your custom templates and tone of voice

This gives you a **compact, specialized, safety-aware technical model** based on Qwen.

---

## **Want a “How to Train” section or example commands?**

I can add:

* Installation instructions
* How to run each training script
* Recommended hardware
* Example training commands
* Inference examples
* A “Contributing” section for collaborators

Just tell me.


---
# Technical Description 

## **📌 Training Pipeline Overview**

This repository implements a complete **LLM fine-tuning pipeline** for Qwen-2.5 models using multiple strategies:

* **Knowledge distillation** (teacher → student)
* **Multi-domain supervised fine-tuning (SFT)**
* **Lightweight domain-focused training (Python + Linux)**
* **Optional LoRA adapters** for low-VRAM setups
* **4-bit quantized model loading** for consumer GPUs

The goal is to produce models that are **more critical, more precise, safer, and more technically reliable** across several domains (Python, Linux, Chat, Excel, Crypto, etc.).

---

## **🧱 1. Data Layer**

The pipeline supports two types of data:

### **A. HuggingFace Datasets**

Depending on the training script, the following datasets are used:

* **openai/openai_humaneval** – Python reasoning and canonical solutions
* **shikhardadhich/linux_commands** – Linux terminal instructions
* **ytz20/LMSYS-Chat-GPT-5-Chat-Response** – Real GPT-5 chat responses
* **WinkingFace/CryptoLM-Bitcoin-BTC-USDT** – Crypto market data
* **Excel formula datasets** – Error correction and formula reasoning
* **NoHallucinations / ArgumentMining** – Critical thinking datasets

These datasets provide **diverse real-world tasks** for the model to learn.

### **B. Synthetic Distillation Data**

In the distillation script, a **larger teacher model** generates high-quality answers for curated prompts.
These responses are saved as `.txt` and later used to train a smaller student model.

---

## **🧹 2. Normalization & Prompt Templates**

All datasets are converted into a **unified one-text-per-line** format containing:

* Domain identification
* User question
* Reference or generated answer
* A system prompt enforcing **critical reasoning**, **error detection**, and **safe behavior**

Example format:

```
[DOMÍNIO: PYTHON] Problema: <problem> Solução de referência: <solution>
```

This step ensures consistency across heterogeneous datasets and reinforces the desired model personality.

---

## **📦 3. Dataset Construction**

After normalization:

1. All texts are merged into a single list
2. A **CausalTextDataset** converts each sample into:

   * `input_ids`
   * `attention_mask`
   * `labels` (same as `input_ids` for causal LM training)
3. A **custom collator** pads inputs correctly to build training batches

Different versions use either:

* `tokenizer.pad`
* or manual padding with `pad_sequence` (more stable across HF versions)

---

## **🧠 4. Model Loading**

All training scripts load **Qwen2.5-1.5B-Instruct** in **4-bit quantization**, enabling fine-tuning on affordable GPUs.

LoRA is supported via:

```python
from peft import LoraConfig, get_peft_model
```

This enables:

* Low VRAM usage
* Fast fine-tuning
* Small, mergeable adapter weights

---

## **⚙️ 5. Training Loop (HF Trainer)**

All models are trained using **HuggingFace Trainer**, with settings such as:

* `optim="paged_adamw_8bit"` for memory efficiency
* Evaluation per epoch
* Optional fp16 / bf16
* Gradient accumulation for larger effective batch sizes
* Saving best model checkpoints

Typical configuration:

```python
TrainingArguments(
    num_train_epochs=1–3,
    per_device_train_batch_size=1–2,
    gradient_accumulation_steps=4–8,
    learning_rate=1e-4 to 2e-4,
    save_strategy="epoch",
)
```

---

## **🎯 6. Output Models**

Depending on the script used, the pipeline yields:

### **A. Distilled Model**

Student model trained on teacher-generated synthetic data.

### **B. Multi-Domain Critical Model**

Trained on Python + Linux + Chat + Crypto (and optionally Excel + critical-thinking datasets).

### **C. Lightweight Model**

Trained only on Python + Linux for fast iterations and limited VRAM.

All models are saved to `output_dir/` along with the tokenizer.

---

## **🔍 Summary Diagram**

```
flowchart TD
    A[Data Sources<br>Python · Linux · Chat · Crypto · Excel] --> B[Normalization & Templates]

    B --> C1[Distillation Pipeline]
    B --> C2[Multi-Domain SFT]
    B --> C3[Light Python+Linux]

    C1 --> D1[Synthetic TXT]
    D1 --> E1[Qwen 4-bit + LoRA]
    E1 --> F1[HF Trainer]
    F1 --> G1[Distilled Model]

    C2 --> E2[Qwen 4-bit + LoRA]
    E2 --> F2[HF Trainer]
    F2 --> G2[Critical Multi-Domain Model]

    C3 --> E3[Qwen 4-bit + LoRA]
    E3 --> F3[HF Trainer]
    F3 --> G3[Light Model]
```

---

## **📘 What This Pipeline Enables**

* Training **full LLMs on consumer GPUs**
* Producing models with **strong critical reasoning**
* Domain-specialized models (Python, Linux, Crypto)
* Experiments with distillation and SFT
* Extremely modular: each domain can be added or removed easily
* Clean dataset architecture suitable for scaling

---




| Script                                 | Synthetic Generation?   | Domains                        | Purpose                      |
| -------------------------------------- | ----------------------- | ------------------------------ | ---------------------------- |
| **train_distilled_llm_qwen_falcon.py** | Yes (teacher → student) | Excel, Python, Arch            | Distillation                 |
| **optimized.py**                       | No                      | Excel, Python, Linux, Critical | Multi-domain SFT             |
| **optimized2.py**                      | No                      | Python, Linux, Chat, Crypto    | Larger multi-domain          |
| **optimized3.py**                      | No                      | Same as v2 (safer JSON)        | Stable version               |
| **critical_multi_domain.py**           | No                      | Python, Linux, Chat, Crypto    | Critical reasoning           |
| **multi_light.py**                     | No                      | Python, Linux                  | Lightweight trainer          |
| **multi_light2.py**                    | No                      | Python, Linux                  | Lightweight + manual padding |
| **python_linux.py**                    | No                      | Python, Linux                  | Minimalist trainer           |
| **python_linux2.py**                   | No                      | Python, Linux                  | Minimalist + manual padding  |





                                      ALL TRAINERS
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────┐
                   │              DATA SOURCES (HF)                   │
                   │  Humaneval | Linux Commands | Excel | Chat | Crypto  │
                   └──────────────────────────────────────────────────┘
                                            │
                                            ▼
                     DATA CLEANING + TEXT UNIFICATION LAYER
                        (normalize_one_line, domain prompts)
                                            │
                          ┌─────────────────┼─────────────────┐
                          │                 │                 │
                          ▼                 ▼                 ▼
        ┌────────────────────────┐ ┌────────────────────────┐ ┌─────────────────────────┐
        │  DISTILLATION FAMILY   │ │ MULTI-DOMAIN FULL      │ │ MULTI-DOMAIN LIGHT      │
        │                        │ │ (Python/Linux/Chat/    │ │ (Python + Linux only)   │
        │ train_distilled_llm_   │ │  Crypto)               │ │                         │
        │ qwen_falcon.py         │ │                        │ │ train_qwen_multi_light.py│
        │                        │ │ train_distilled_llm_   │ │ train_qwen_multi_light2.py│
        │ (Teacher Model →       │ │ qwen_optimized.py       │ │ train_qwen_python_linux.py│
        │  Synthetic Data →      │ │ train_distilled_llm_    │ │ train_qwen_python_linux2.py│
        │  Student Fine-Tune)    │ │ qwen_optimized2.py      │ └─────────────────────────┘
        └────────────────────────┘ │ train_distilled_llm_    │
                 │                 │ qwen_optimized3.py      │
                 │                 │ train_qwen_critical_     │
                 │                 │ multi_domain.py          │
                 ▼                 └────────────────────────┘
        SYNTHETIC DATA (TXT)
                 │
                 ▼
   ┌───────────────────────────┐
   │ SyntheticTextDataset      │
   │  + Collator               │  (shared structure across all families)
   └───────────────────────────┘
                 │
                 ▼
       TOKENIZER + QWEN MODEL LOAD
       (4-bit quantization, optional LoRA)
                 │
                 ▼
         ┌───────────────────────────┐
         │ HF Trainer (train/eval)  │
         │ eval per epoch or steps  │
         │ bf16/fp16, 8-bit optim   │
         └───────────────────────────┘
                 │
                 ▼
         SAVED MODEL OUTPUT DIR

