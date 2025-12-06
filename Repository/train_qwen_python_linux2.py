# -*- coding: utf-8 -*-
"""
Treino crítico multi-domínio (PYTHON + LINUX) em Qwen2.5-1.5B-Instruct.

- Carrega:
    * openai/openai_humaneval  (problemas + soluções em Python)
    * shikhardadhich/linux_commands  (instruções -> comandos Linux)
- Converte tudo para texto único por amostra (causal LM).
- Tokeniza em __getitem__ e usa collator com padding manual (pad_sequence).
- Treina Qwen em 4-bit + LoRA (opcional).
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

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


# =========================================================
# 1. Dataset de texto causal (já tokenizado por amostra)
# =========================================================
class CausalTextDataset(Dataset):
    """
    Recebe uma lista de strings e gera (input_ids == labels) por amostra.
    Nenhum campo extra é retornado (evita erros de padding).
    """

    def __init__(self, tokenizer, texts, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.max_length = max_length

        if len(self.texts) == 0:
            raise ValueError("Nenhum texto encontrado para o dataset.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding só no collator
            return_tensors="pt",
        )
        # remove dimensão de batch e garante dtype long
        item = {k: v.squeeze(0).to(dtype=torch.long) for k, v in enc.items()}
        # labels = input_ids (treino causal)
        item["labels"] = item["input_ids"].clone()
        return item


# =========================================================
# 2. Data collator com padding manual
# =========================================================
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Faz o padding manual usando pad_sequence, sem passar pelo tokenizer.pad
        (evita problemas de versões antigas do transformers).
        """
        # cada elemento em features é um dict com tensores 1D: input_ids, attention_mask, labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,  # 0 = posição de padding
        )

        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,  # -100 = ignorado na loss
        )

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
        return batch


# =========================================================
# 3. Formatação crítica dos domínios
# =========================================================

def normalize_one_line(text: str) -> str:
    """Colapsa espaços/quebras de linha para evitar textos gigantes em várias linhas."""
    return " ".join(str(text).split())


def build_python_text(problem: str, solution: str) -> str:
    header = (
        "Você é uma assistente de programação em Python, extremamente crítica e cuidadosa. "
        "Seu objetivo é analisar o código, apontar más práticas, erros lógicos e sugerir "
        "correções claras. Não aceite soluções erradas sem questionar.\n\n"
    )
    body = (
        "[DOMÍNIO: PYTHON]\n"
        "Problema:\n"
        f"{problem}\n\n"
        "Solução de referência (pode conter problemas que devem ser criticados e melhorados):\n"
        f"{solution}\n"
    )
    return normalize_one_line(header + body)


def build_linux_text(instruction: str, given_input: str, command: str) -> str:
    header = (
        "Você é uma assistente de linha de comando em Linux (Arch-like), extremamente crítica "
        "com segurança e boas práticas. Aponte perigos de rodar comandos como root, uso "
        "indevido de flags e alternativas mais seguras quando necessário.\n\n"
    )
    context = instruction or ""
    if given_input:
        context += f"\nContexto adicional:\n{given_input}"

    body = (
        "[DOMÍNIO: LINUX]\n"
        "Pedido do usuário:\n"
        f"{context}\n\n"
        "Comando proposto como resposta correta:\n"
        f"{command}\n"
    )
    return normalize_one_line(header + body)


# =========================================================
# 4. Carregamento dos datasets HF (Python + Linux)
# =========================================================

def load_python_datasets(limit_per_domain: Optional[int] = None):
    """
    openai/openai_humaneval -> usamos prompt + canonical_solution.
    """
    print("Carregando dataset Python: openai/openai_humaneval ...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    def _format(example):
        problem = example.get("prompt", "")
        solution = (
            example.get("canonical_solution")
            or example.get("solution")
            or ""
        )
        return {"text": build_python_text(problem, solution)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit_per_domain is not None and limit_per_domain > 0:
        limit = min(limit_per_domain, ds.num_rows)
        ds = ds.select(range(limit))
    return ds


def load_linux_datasets(limit_per_domain: Optional[int] = None):
    """
    shikhardadhich/linux_commands -> columns: Instruction, Input, Output (ou similares).
    """
    print("Carregando dataset Linux: shikhardadhich/linux_commands ...")
    ds = load_dataset("shikhardadhich/linux_commands", split="train")

    cols = ds.column_names
    instr_col_candidates = ["Instruction", "instruction", "prompt", "Prompt", "question"]
    input_col_candidates = ["Input", "input", "context", "Context"]
    output_col_candidates = ["Output", "output", "command", "Command", "completion"]

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    instr_col = pick(instr_col_candidates)
    out_col = pick(output_col_candidates)
    in_col = pick(input_col_candidates)

    if instr_col is None or out_col is None:
        raise ValueError(
            f"Colunas inesperadas em shikhardadhich/linux_commands: {cols} "
            f"(não encontrei Instruction/Output equivalentes)."
        )

    def _format(example):
        instr = example.get(instr_col, "")
        inp = example.get(in_col, "") if in_col is not None else ""
        cmd = example.get(out_col, "")
        return {"text": build_linux_text(instr, inp, cmd)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit_per_domain is not None and limit_per_domain > 0:
        limit = min(limit_per_domain, ds.num_rows)
        ds = ds.select(range(limit))
    return ds


def build_full_text_dataset(tokenizer, max_length: int, limit_per_domain: Optional[int] = None):
    """
    Junta todos os domínios em um único Dataset pytorch-ready.
    """
    ds_py = load_python_datasets(limit_per_domain)
    ds_linux = load_linux_datasets(limit_per_domain)

    print(f"Python samples: {ds_py.num_rows} | Linux samples: {ds_linux.num_rows}")

    combined = concatenate_datasets([ds_py, ds_linux]).shuffle(seed=42)
    texts = combined["text"]
    return CausalTextDataset(tokenizer=tokenizer, texts=texts, max_length=max_length)


# =========================================================
# 5. CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning crítico multi-domínio (Python + Linux) em Qwen2.5-1.5B-Instruct"
    )

    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Ex: Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Diretório de saída para o modelo fine-tunado.",
    )

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--limit_per_domain",
        type=int,
        default=200,
        help="Máximo de exemplos por domínio (Python / Linux). Use valores menores para testes.",
    )

    # Treino
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_ratio", type=float, default=0.05)

    return parser.parse_args()


# =========================================================
# 6. Main
# =========================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    print("Carregando tokenizer do modelo aluno...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset combinado
    print("Construindo dataset multi-domínio (PYTHON + LINUX)...")
    full_dataset = build_full_text_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit_per_domain=args.limit_per_domain,
    )

    n_total = len(full_dataset)
    n_eval = max(1, int(n_total * args.eval_ratio))
    n_train = n_total - n_eval

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_eval]
    )
    print(f"Total exemplos: {n_total} | treino: {n_train} | val: {n_eval}")

    # Modelo aluno em 4-bit
    print(f"Carregando modelo aluno para treino: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    # LoRA opcional
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

    # Argumentos de treino — usando eval_strategy (compatível com HF antigo)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="epoch",          # <- chave compatível com sua versão
        fp16=args.fp16,
        bf16=args.bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
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

    print("Iniciando treinamento...")
    trainer.train()

    print("Salvando modelo fine-tunado...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Treino concluído.")


if __name__ == "__main__":
    main()
