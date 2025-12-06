# -*- coding: utf-8 -*-
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union

import random

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


# ===========================
# 1. Dataset (1 linha = 1 exemplo) + limpeza
# ===========================
class SyntheticTextDataset(Dataset):
    """
    Espera arquivo .txt com 1 exemplo por linha.
    Cada linha é um exemplo de linguagem causal (input_ids == labels).
    O construtor faz limpeza básica para evitar linhas quebradas.
    """

    def __init__(self, tokenizer, data_path: str, max_length: int = 512):
        assert os.path.exists(data_path), f"Arquivo de dados não encontrado: {data_path}"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[str] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                # descarta linhas vazias
                if not line:
                    continue
                # descarta linhas muito curtas (quase sempre ruído)
                if len(line) < 10:
                    continue
                # descarta linhas que parecem JSON/arrays crus
                if (line.startswith("[") and line.endswith("]")) or (
                    line.startswith("{") and line.endswith("}")
                ):
                    continue

                # tentativa de tokenização para garantir que não quebra depois
                try:
                    _ = tokenizer(
                        line,
                        truncation=True,
                        max_length=self.max_length,
                    )
                except Exception:
                    # se der erro na tokenização, pula a linha
                    continue

                self.examples.append(line)

        if len(self.examples) == 0:
            raise ValueError("Arquivo de dados está vazio ou todas as linhas foram descartadas na limpeza.")

        print(f"[SyntheticTextDataset] {len(self.examples)} exemplos válidos após limpeza.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Retorna dicionário no formato esperado pelo data_collator:
        listas de inteiros (não tensores), para o tokenizer.pad tratar.
        """
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # não usamos padding aqui, o collator fará isso
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        # labels = mesma sequência que input_ids
        item["labels"] = enc["input_ids"].copy()
        return item


# ===========================
# 2. Data Collator seguro
# ===========================
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Usa tokenizer.pad para padronizar tamanhos de input_ids, attention_mask e labels.
        Garante que 'labels' também sejam tensores de mesmo comprimento.
        """
        # tokenizer.pad vai cuidar de input_ids / attention_mask;
        # ensure labels are list[int] (não tensores) antes do pad
        for f in features:
            if isinstance(f.get("labels"), torch.Tensor):
                f["labels"] = f["labels"].tolist()

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )

        # por segurança, converte labels explicitamente para long
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(dtype=torch.long)

        return batch


# ===========================
# 3. Prompts para geração crítica
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
    Prompt em português instruindo o modelo a ser crítico, questionador
    e focado em um domínio específico.
    """
    header = (
        "Você é uma assistente técnica extremamente crítica e cuidadosa. "
        "Seu objetivo é apontar erros, más práticas e premissas equivocadas, "
        "explicando claramente o porquê. Você nunca deve apenas concordar: "
        "se algo estiver errado, diga que está errado e corrija. "
        "Se não tiver informação suficiente, diga isso explicitamente.\n\n"
    )

    domain_line = f"[DOMÍNIO: {domain.upper()}]\n"
    conversation = f"Usuário: {user_question}\nAssistente:"
    return header + domain_line + conversation


def extract_answer_from_generated(full_text: str, prompt: str) -> str:
    """
    Tenta extrair somente a resposta após 'Assistente:'.
    Se não encontrar, remove o prompt bruto do começo.
    """
    idx = full_text.find("Assistente:")
    if idx != -1:
        answer = full_text[idx + len("Assistente:"):].strip()
    else:
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()
    # normaliza em uma linha
    answer = " ".join(answer.split())
    return answer


def generate_synthetic_dataset(
    teacher_model,
    tokenizer,
    output_path: str,
    samples_per_domain: int = 50,
    max_new_tokens: int = 128,
    batch_size_gen: int = 4,
):
    """
    Gera base sintética contendo:
      - Excel
      - Python
      - Arch Linux
    com respostas críticas do modelo professor.
    Geração em lote (batch) para aproveitar melhor a GPU.
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

                for prompt_str, output_ids, user_question in zip(prompts, gen_outputs, user_questions):
                    full_text = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                    )
                    answer = extract_answer_from_generated(full_text, prompt_str)
                    line = f"[{domain_name}] Usuário: {user_question} Assistente: {answer}"
                    line = " ".join(line.split())
                    f_out.write(line + "\n")

                i += current_batch_size
                print(f"  {i}/{n} exemplos gerados para {domain_name}")

    print(f"Base sintética gerada em: {output_path}")


# ===========================
# 4. CLI & main
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Treino de Qwen2.5-1.5B com geração sintética crítica (professor 7B) e fine-tuning"
    )

    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Modelo professor usado só para gerar dados.",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Ex: Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--synthetic_data",
        type=str,
        required=True,
        help="Arquivo .txt para dataset sintético (será criado/reescrito se --generate_synthetic).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Diretório de saída para o modelo aluno fine-tunado.",
    )
    parser.add_argument("--max_length", type=int, default=512)

    # Treino
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

    # Geração sintética
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Se passado, gera a base sintética antes do treino.",
    )
    parser.add_argument(
        "--samples_per_domain",
        type=int,
        default=50,
        help="Quantos exemplos gerar por domínio (Excel, Python, Arch).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Máximo de tokens novos gerados por resposta sintética.",
    )
    parser.add_argument(
        "--batch_size_gen",
        type=int,
        default=4,
        help="Tamanho de batch na geração sintética.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Tokenizer do aluno
    print("Carregando tokenizer do modelo aluno...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Geração sintética (apenas professor)
    if args.generate_synthetic:
        print(f"Carregando modelo professor (geração): {args.teacher_model}")
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

        # libera professor da GPU/CPU
        del teacher_model
        torch.cuda.empty_cache()

    # 3) Dataset de treino/val com limpeza
    print("Carregando dataset sintético para treino...")
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

    # 4) Modelo aluno (1.5B) com 4-bit + (opcional) LoRA
    print(f"Carregando modelo aluno (treino): {args.student_model}")
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

    # 5) Configuração de treino (com eval_strategy suportado nas versões novas)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # 6) Trainer padrão
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

    print("Salvando modelo aluno fine-tunado...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Treino finalizado.")


if __name__ == "__main__":
    main()
