# IPMI
LLM specilized on social security legislation
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

