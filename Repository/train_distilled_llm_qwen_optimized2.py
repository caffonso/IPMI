import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# LoRA opcional
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import random


# ===========================
# 1. Dataset (1 linha = 1 exemplo)
# ===========================
class SyntheticTextDataset(Dataset):
    """
    Espera arquivo .txt com 1 exemplo por linha.
    Cada linha é um exemplo de linguagem causal (input_ids == labels).
    """

    def __init__(self, tokenizer, data_path: str, max_length: int = 512):
        assert os.path.exists(data_path), f"Arquivo de dados não encontrado: {data_path}"
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = [l.strip() for l in f if l.strip()]

        if len(self.examples) == 0:
            raise ValueError("Arquivo de dados está vazio.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


# ===========================
# 2. Data Collator
# ===========================
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )
        return batch


# ===========================
# 3. Queries críticas
# ===========================
EXCEL_QUERIES = [
    "Preciso de uma fórmula de Excel que some apenas os valores positivos em uma coluna, mas acho que =SOMA.SE(A1:A100,\">0\") resolve. Está certo?",
    "Quero contar apenas células não vazias em um intervalo, estou usando =SOMA(A1:A10<>\"\"). Explique se essa fórmula faz sentido e corrija se estiver errada.",
    "Quero procurar um valor em uma tabela, mas odeio PROCV. Existe alguma alternativa mais robusta? Critique minha ideia de usar PROCV em tudo.",
    "Estou misturando referência absoluta e relativa em uma fórmula complicada de planilha. Explique quando usar $A$1 versus A1 em um contexto real.",
    "Tenho uma planilha que está extremamente lenta. Quais erros comuns nas fórmulas devo verificar antes de culpar o hardware?",
]

PYTHON_QUERIES = [
    "Esse código em Python está certo? \nfor i in range(10):\n    print(i)\n    i += 1\nSe não estiver, critique e explique o problema conceitual.",
    "Estou usando try/except para esconder todos os erros com 'except Exception as e: pass'. Isso é boa prática? Seja bem crítico.",
    "Quero uma função em Python que leia um arquivo grande inteiro com read(). Isso é uma má ideia? Quando e por que devo evitar?",
    "Escrevi uma função recursiva sem condição de parada clara porque 'na prática funciona'. Explique por que isso é perigoso.",
    "Estou misturando lógica de negócio com código de I/O no mesmo bloco de função. Critique meu design e sugira melhorias.",
]

ARCH_QUERIES = [
    "No Arch Linux, estou rodando tudo como root porque é mais simples. Explique, de forma crítica, por que isso é um problema de segurança.",
    "Eu removi pacotes com 'pacman -Rdd' só para resolver conflitos. Quais erros conceituais estou cometendo?",
    "Quero editar arquivos em /etc diretamente sem usar versionamento ou backup. Critique essa abordagem e sugira algo melhor.",
    "Estou usando 'sudo pacman -Sy' sem '-u' porque é mais rápido. Aponte todos os problemas dessa prática.",
    "Quero compilar tudo do AUR sem ler os PKGBUILDs porque confio na comunidade. Explique por que isso é uma má ideia.",
]


def build_critical_system_prompt(domain: str, user_question: str) -> str:
    """
    Monta prompt para gerar respostas críticas.
    """
    header = (
        "Você é uma assistente técnica extremamente crítica e cuidadosa. "
        "Seu objetivo é apontar erros, más práticas e premissas equivocadas, "
        "explicando claramente o porquê. Você nunca deve apenas concordar — "
        "se algo estiver errado, diga que está errado e corrija. "
        "Se não tiver informação suficiente, diga isso explicitamente.\n\n"
    )

    domain_line = f"[DOMÍNIO: {domain.upper()}]\n"
    conversation = f"Usuário: {user_question}\nAssistente:"

    return header + domain_line + conversation


def extract_answer_from_generated(full_text: str, prompt: str) -> str:
    """
    Extrai apenas o texto após 'Assistente:'.
    """
    idx = full_text.find("Assistente:")
    if idx != -1:
        answer = full_text[idx + len("Assistente:"):].strip()
    else:
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()

    return " ".join(answer.split())


def generate_synthetic_dataset(
    teacher_model,
    tokenizer,
    output_path: str,
    samples_per_domain: int = 50,
    max_new_tokens: int = 128,
    batch_size_gen: int = 4,
):
    """
    Gera dataset sintético crítico para Excel, Python e Arch Linux.
    Utiliza o modelo professor.
    """
    device = next(teacher_model.parameters()).device

    all_domains = [
        ("EXCEL", EXCEL_QUERIES),
        ("PYTHON", PYTHON_QUERIES),
        ("ARCH", ARCH_QUERIES),
    ]

    teacher_model.eval()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for domain_name, query_list in all_domains:
            print(f"Gerando exemplos para domínio: {domain_name}")
            n = samples_per_domain
            i = 0

            while i < n:
                current_batch_size = min(batch_size_gen, n - i)
                user_questions = [random.choice(query_list) for _ in range(current_batch_size)]

                prompts = [
                    build_critical_system_prompt(domain_name, q)
                    for q in user_questions
                ]

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    gen_outputs = teacher_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                for prompt_str, output_ids, user_question in zip(
                    prompts, gen_outputs, user_questions
                ):
                    full_text = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                    )
                    answer = extract_answer_from_generated(full_text, prompt_str)

                    line = (
                        f"[{domain_name}] Usuário: {user_question} Assistente: {answer}"
                    )
                    line = " ".join(line.split())
                    f_out.write(line + "\n")

                i += current_batch_size
                print(f"  {i}/{n} exemplos gerados para {domain_name}")

    print(f"Base sintética gerada em: {output_path}")


# ===========================
# 4. CLI
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Treino de Qwen2.5-1.5B com geração sintética crítica."
    )

    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--synthetic_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_ratio", type=float, default=0.05)

    parser.add_argument("--generate_synthetic", action="store_true")
    parser.add_argument("--samples_per_domain", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size_gen", type=int, default=4)

    return parser.parse_args()


# ===========================
# 5. main()
# ===========================
def main():
    args = parse_args()

    print("Carregando tokenizer do modelo aluno...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.generate_synthetic:
        print(f"Carregando modelo professor: {args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        generate_synthetic_dataset(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            output_path=args.synthetic_data,
            samples_per_domain=args.samples_per_domain,
            max_new_tokens=args.max_new_tokens,
            batch_size_gen=args.batch_size_gen,
        )

        del teacher_model
        torch.cuda.empty_cache()

    print("Carregando dataset sintético...")
    full_dataset = SyntheticTextDataset(
        tokenizer=tokenizer,
        data_path=args.synthetic_data,
        max_length=args.max_length,
    )

    n_total = len(full_dataset)
    n_eval = max(1, int(n_total * args.eval_ratio))
    n_train = n_total - n_eval

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_eval]
    )

    print(f"Total exemplos: {n_total} | treino: {n_train} | val: {n_eval}")

    print(f"Carregando modelo aluno: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft não está instalado. Rode: pip install peft")

        print("Aplicando LoRA ao modelo aluno...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        max_steps=200,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=args.fp16,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        report_to="none",
    )

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Iniciando treinamento do aluno...")
    trainer.train()

    print("Salvando modelo fine-tunado...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Treino finalizado.")


if __name__ == "__main__":
    main()

