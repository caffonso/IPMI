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
