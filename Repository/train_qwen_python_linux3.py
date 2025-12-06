# -*- coding: utf-8 -*-
"""
Treino crítico multi-domínio (Python + Linux + GPT5 Chat + SYNTH + Python Alpaca)
em Qwen2.5-1.5B-Instruct.

Datasets incluídos:
- openai/openai_humaneval
- shikhardadhich/linux_commands
- ytz20/LMSYS-Chat-GPT-5-Chat-Response
- PleIAs/SYNTH
- iamtarun/python_code_instructions_18k_alpaca

O modelo aluno é treinado usando causal LM (input_ids == labels).
O collator usa pad_sequence manual, evitando erros de batch.
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
# 1. Normalização
# =========================================================

def normalize_one_line(text: str) -> str:
    return " ".join(str(text).split())


# =========================================================
# 2. Dataset base
# =========================================================

class CausalTextDataset(Dataset):
    """
    Recebe lista de strings -> retorna input_ids, attention_mask e labels.
    """

    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.max_length = max_length

        if len(self.texts) == 0:
            raise ValueError("Dataset final vazio!")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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


# =========================================================
# 3. Data collator com pad_sequence
# =========================================================

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
# 4. Formatação crítica por domínio
# =========================================================

def build_python_text(problem: str, solution: str) -> str:
    header = (
        "Você é uma assistente de programação em Python, extremamente crítica. "
        "Aponte erros, más práticas e sugeria melhorias.\n\n"
        "[DOMÍNIO: PYTHON]\n"
    )
    return normalize_one_line(f"{header}Problema: {problem}\nResposta: {solution}")


def build_linux_text(instr: str, ginput: str, command: str) -> str:
    header = (
        "Você é uma assistente de Linux/terminal extremamente crítica com segurança. "
        "[DOMÍNIO: LINUX]\n"
    )
    full = f"{header}Instrução: {instr}\nContexto: {ginput}\nComando: {command}"
    return normalize_one_line(full)


# =========================================================
# 5. Loaders dos datasets
# =========================================================

def load_python_humaneval(limit=None):
    print("Carregando openai/openai_humaneval...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    def _f(e):
        p = e.get("prompt", "")
        s = e.get("canonical_solution") or e.get("solution", "")
        return {"text": build_python_text(p, s)}

    ds = ds.map(_f, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))

    return ds


def load_linux_commands(limit=None):
    print("Carregando shikhardadhich/linux_commands...")
    ds = load_dataset("shikhardadhich/linux_commands", split="train")
    cols = ds.column_names

    def pick(names):
        for n in names:
            if n in cols:
                return n
        return None

    instr = pick(["Instruction", "instruction", "prompt"])
    inp = pick(["Input", "input", "context"])
    out = pick(["Output", "output", "command"])

    if instr is None or out is None:
        raise ValueError(f"Colunas desconhecidas no dataset Linux: {cols}")

    def _f(e):
        i = e.get(instr, "")
        c = e.get(out, "")
        gi = e.get(inp, "")
        return {"text": build_linux_text(i, gi, c)}

    ds = ds.map(_f, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))

    return ds


def load_chat_gpt5(limit=None):
    HF_ID = "ytz20/LMSYS-Chat-GPT-5-Chat-Response"
    print(f"Carregando {HF_ID}...")
    ds = load_dataset(HF_ID, split="train")

    def _f(e):
        instr = e.get("instruction", "")
        resp = e.get("response", "")
        text = f"[CHAT] Usuário: {instr}\nAssistente: {resp}"
        return {"text": normalize_one_line(text)}

    ds = ds.map(_f, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))

    return ds


def load_synth(limit=None):
    HF_ID = "PleIAs/SYNTH"
    print(f"Carregando {HF_ID}...")
    ds = load_dataset(HF_ID, split="train")

    def _f(e):
        p = e.get("prompt", "")
        c = e.get("completion", "")
        t = f"[SYNTH] Usuário: {p}\nAssistente: {c}"
        return {"text": normalize_one_line(t)}

    ds = ds.map(_f, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))

    return ds


def load_python_alpaca(limit=None):
    HF_ID = "iamtarun/python_code_instructions_18k_alpaca"
    print(f"Carregando {HF_ID}...")
    ds = load_dataset(HF_ID, split="train")

    def _f(e):
        instr = e.get("instruction", "")
        inp = e.get("input", "")
        out = e.get("output", "")
        user_prompt = instr.strip()
        if inp:
            user_prompt += " " + inp.strip()
        t = f"[PYTHON-INSTR] Usuário: {user_prompt}\nAssistente: {out}"
        return {"text": normalize_one_line(t)}

    ds = ds.map(_f, remove_columns=ds.column_names)

    if limit:
        ds = ds.shuffle(42).select(range(min(limit, ds.num_rows)))

    return ds


# =========================================================
# 6. Constru�ão do dataset completo
# =========================================================

def build_full_dataset(tokenizer, max_length, limit):
    ds_py = load_python_humaneval(limit)
    ds_linux = load_linux_commands(limit)
    ds_chat = load_chat_gpt5(limit)
    ds_synth = load_synth(limit)
    ds_py_alp = load_python_alpaca(limit)

    print(
        f"Humaneval: {ds_py.num_rows} | Linux: {ds_linux.num_rows} | "
        f"GPT5: {ds_chat.num_rows} | SYNTH: {ds_synth.num_rows} | "
        f"Python Alpaca: {ds_py_alp.num_rows}"
    )

    combined = concatenate_datasets(
        [ds_py, ds_linux, ds_chat, ds_synth, ds_py_alp]
    ).shuffle(42)

    return CausalTextDataset(
        tokenizer=tokenizer,
        texts=combined["text"],
        max_length=max_length,
    )


# =========================================================
# 7. CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--student_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--limit_per_domain", type=int, default=200)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_lora", action="store_true")

    return p.parse_args()


# =========================================================
# 8. MAIN
# =========================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Construindo dataset completo...")
    dataset = build_full_dataset(
        tokenizer, args.max_length, args.limit_per_domain
    )

    n_total = len(dataset)
    n_eval = max(1, int(n_total * 0.05))
    n_train = n_total - n_eval

    train_ds, eval_ds = torch.utils.data.random_split(dataset, [n_train, n_eval])

    print(f"Treino: {n_train} | Val: {n_eval}")

    print("Carregando modelo aluno...")
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
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )

    print("Treinando...")
    trainer.train()

    print("Salvando...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Treino concluído.")


if __name__ == "__main__":
    main()
