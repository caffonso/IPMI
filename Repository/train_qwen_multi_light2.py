# -*- coding: utf-8 -*-
"""
Treino crítico multi-domínio (Python + Linux + Chat + Crypto) em Qwen2.5-1.5B-Instruct.

Datasets usados:
- openai/openai_humaneval          (pequeno, código Python)
- shikhardadhich/linux_commands    (médio, comandos Linux)
- ytz20/LMSYS-Chat-GPT-5-Chat-Response  (prompts reais + resposta GPT-5-Chat)
- WinkingFace/CryptoLM-Bitcoin-BTC-USDT (dados de mercado BTC/USDT, amostrados)

Objetivos:
- Tornar o modelo mais crítico e técnico
- Aprender padrões de Python, Linux, chat geral e contexto de cripto BTC/USDT
"""

import os
import json
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
# Helpers
# =========================================================

def normalize_one_line(text: str) -> str:
    """Colapsa quebras de linha e múltiplos espaços em uma única linha."""
    return " ".join(str(text).split())


class CausalTextDataset(Dataset):
    """Lista de strings -> input_ids / attention_mask / labels."""

    def __init__(self, tokenizer, texts, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.max_length = max_length
        if not self.texts:
            raise ValueError("Dataset final vazio!")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0).to(torch.long) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =========================================================
# Formatação crítica por domínio
# =========================================================

def build_python_text(problem: str, solution: str) -> str:
    header = (
        "Você é uma assistente de programação em Python, extremamente crítica. "
        "Aponte erros conceituais, más práticas e explique melhorias.\n\n"
        "[DOMÍNIO: PYTHON]\n"
    )
    return normalize_one_line(
        f"{header}Problema: {problem}\nSolução de referência: {solution}"
    )


def build_linux_text(instr: str, ginput: str, command: str) -> str:
    header = (
        "Você é uma assistente de Linux/terminal extremamente crítica em segurança e boas práticas.\n"
        "[DOMÍNIO: LINUX]\n"
    )
    full = f"{header}Pergunta do usuário: {instr}\nContexto: {ginput}\nComando recomendado: {command}"
    return normalize_one_line(full)


def build_chat_text(prompt: str, answer: str) -> str:
    header = (
        "Você é uma assistente de chat, mas não concorda automaticamente com o usuário. "
        "Você aponta problemas, inconsistências e pede clareza quando necessário.\n"
        "[DOMÍNIO: CHAT-GPT5]\n"
    )
    full = f"{header}Usuário: {prompt}\nAssistente (GPT-5): {answer}"
    return normalize_one_line(full)


def build_crypto_text(row: Dict) -> str:
    """
    Converte uma linha tabular de BTC/USDT em texto.
    Corrige o erro: Timestamps não são JSON serializáveis.
    Converte todos os valores não-serializáveis em string.
    """

    safe_row = {}
    for k, v in row.items():
        # Converte Timestamps, numpy datetimes, etc.
        if isinstance(v, (int, float, str)) or v is None:
            safe_row[k] = v
        else:
            # fallback: tudo vira string segura
            safe_row[k] = str(v)

    json_row = json.dumps(safe_row, ensure_ascii=False)

    text = (
        "Você é uma assistente técnica em trading de criptomoedas (BTC/USDT). "
        "Descreva padrões com cuidado, não faça promessas de lucro, "
        "e aponte riscos de forma crítica.\n"
        "[DOMÍNIO: CRYPTO BTC-USDT]\n"
        f"DADOS: {json_row}"
    )
    return normalize_one_line(text)

# =========================================================
# Loaders de datasets
# =========================================================

def load_python_humaneval(limit: Optional[int] = None):
    print("Carregando openai/openai_humaneval...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    def _format(e):
        p = e.get("prompt", "")
        s = e.get("canonical_solution") or e.get("solution", "")
        return {"text": build_python_text(p, s)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))
    return ds


def load_linux_commands(limit: Optional[int] = None):
    print("Carregando shikhardadhich/linux_commands...")
    ds = load_dataset("shikhardadhich/linux_commands", split="train")
    cols = ds.column_names

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    instr_col = pick(["Instruction", "instruction", "prompt"])
    input_col = pick(["Input", "input", "context"])
    output_col = pick(["Output", "output", "command"])

    if instr_col is None or output_col is None:
        raise ValueError(f"Colunas inesperadas em linux_commands: {cols}")

    def _format(e):
        instr = e.get(instr_col, "") or ""
        inp = e.get(input_col, "") or ""
        out = e.get(output_col, "") or ""
        return {"text": build_linux_text(instr, inp, out)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))
    return ds


def load_lmsys_gpt5(limit: Optional[int] = None):
    """
    ytz20/LMSYS-Chat-GPT-5-Chat-Response
    Colunas: content (prompt), teacher_response (resposta GPT-5-Chat)
    Split principal: train
    """
    print("Carregando ytz20/LMSYS-Chat-GPT-5-Chat-Response...")
    ds = load_dataset("ytz20/LMSYS-Chat-GPT-5-Chat-Response", split="train")

    def _format(e):
        prompt = e.get("content", "") or ""
        answer = e.get("teacher_response", "") or ""
        return {"text": build_chat_text(prompt, answer)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))
    return ds


def load_crypto_btc_usdt(limit: Optional[int] = None, month_split: Optional[str] = None):
    """
    Carrega WinkingFace/CryptoLM-Bitcoin-BTC-USDT.

    Se month_split for None -> usa split 'train' completo.
    Caso contrário, usa fatia específica: ex: 'train[2024-01.parquet]'.
    """
    print("Carregando WinkingFace/CryptoLM-Bitcoin-BTC-USDT...")

    if month_split is None:
        ds = load_dataset("WinkingFace/CryptoLM-Bitcoin-BTC-USDT", split="train")
    else:
        # Atenção: isso assume que o dataset define splits/filtros com essa sintaxe.
        ds = load_dataset(
            "WinkingFace/CryptoLM-Bitcoin-BTC-USDT",
            split="train",  # mantém train e filtra depois se necessário
        )

    def _format(e):
        # e pode ser LazyRow -> garante dict antes de passar para build_crypto_text
        row = dict(e)
        return {"text": build_crypto_text(row)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))
    return ds


def build_full_dataset(
    tokenizer,
    max_length: int,
    limit_per_domain: int,
    crypto_month_split: Optional[str] = None,
):
    """
    Constrói o dataset combinado com todos os domínios.

    limit_per_domain controla quantos exemplos de CADA dataset entram.
    """
    ds_py = load_python_humaneval(limit_per_domain)
    ds_linux = load_linux_commands(limit_per_domain)
    ds_chat = load_lmsys_gpt5(limit_per_domain)
    ds_crypto = load_crypto_btc_usdt(
        limit=limit_per_domain,
        month_split=crypto_month_split,
    )

    print(
        f"Humaneval: {ds_py.num_rows} | Linux: {ds_linux.num_rows} | "
        f"LMSYS-GPT5: {ds_chat.num_rows} | Crypto: {ds_crypto.num_rows}"
    )

    combined = concatenate_datasets(
        [ds_py, ds_linux, ds_chat, ds_crypto]
    ).shuffle(42)

    return CausalTextDataset(
        tokenizer=tokenizer,
        texts=combined["text"],
        max_length=max_length,
    )


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning crítico multi-domínio em Qwen2.5-1.5B-Instruct"
    )
    p.add_argument(
        "--student_model",
        required=True,
        help="Ex: Qwen/Qwen2.5-1.5B-Instruct",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)

    # Controle de tamanho
    p.add_argument(
        "--limit_per_domain",
        type=int,
        default=200,
        help="Máximo de exemplos por dataset (Humaneval, Linux, LMSYS, Crypto).",
    )

    # Crypto split (opcional)
    p.add_argument(
        "--crypto_month_split",
        type=str,
        default=None,
        help="Split do CryptoLM a usar (ex: 2024-01.parquet). Se None, usa o train completo.",
    )

    # Treino
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    return p.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Construindo dataset completo...")
    dataset = build_full_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit_per_domain=args.limit_per_domain,
        crypto_month_split=args.crypto_month_split,
    )

    n_total = len(dataset)
    n_eval = max(1, int(n_total * 0.05))
    n_train = n_total - n_eval
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [n_train, n_eval])
    print(f"Treino: {n_train} | Val: {n_eval}")

    print("Carregando modelo aluno (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("Instale peft: pip install peft")
        print("Aplicando LoRA...")
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        evaluation_strategy="epoch",   # nome correto do argumento
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Iniciando treino...")
    trainer.train()

    print("Salvando modelo...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Treino concluído.")


if __name__ == "__main__":
    main()
