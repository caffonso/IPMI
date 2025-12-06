import os
import random
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch

# LoRA
from peft import LoraConfig, get_peft_model


# ============================================================
# Helpers
# ============================================================

def normalize_pair(prompt, answer):
    """Normaliza qualquer par de entrada/saída."""
    if prompt is None or answer is None:
        return None
    p = str(prompt).strip()
    a = str(answer).strip()
    if len(p) < 3 or len(a) < 3:
        return None
    return {
        "prompt": p,
        "response": a
    }


def load_excel_datasets(limit):
    print("Carregando datasets Excel...")

    # 1. excel-formula-correction (perfeito p/ crítica)
    d1 = load_dataset("Ayushk4/excel-formula-correction", split="train")
    excel1 = []
    for x in d1:
        pair = normalize_pair(x.get("incorrect_formula"), x.get("correct_formula"))
        if pair:
            excel1.append(pair)

    # 2. excelformer
    d2 = load_dataset("jyansir/excelformer", split="train")
    excel2 = []
    for x in d2:
        pair = normalize_pair(x.get("prompt"), x.get("formula"))
        if pair:
            excel2.append(pair)

    combined = excel1 + excel2
    random.shuffle(combined)
    return combined[:limit]


def load_python_datasets(limit):
    print("Carregando datasets Python...")

    # 1. python-error-correction
    d1 = load_dataset("fblaese/python-error-correction", split="train")
    py1 = []
    for x in d1:
        pair = normalize_pair(x.get("buggy_code"), x.get("fixed_code"))
        if pair:
            py1.append(pair)

    # 2. PythonCodeInstruct
    d2 = load_dataset("TheRamU/PythonCodeInstruct", split="train")
    py2 = []
    for x in d2:
        pair = normalize_pair(x.get("instruction"), x.get("output"))
        if pair:
            py2.append(pair)

    combined = py1 + py2
    random.shuffle(combined)
    return combined[:limit]


def load_linux_datasets(limit):
    print("Carregando datasets Linux / Bash...")

    # 1. BashScriptFix
    d1 = load_dataset("SUN-L/BashScriptFix", split="train")
    bash1 = []
    for x in d1:
        pair = normalize_pair(x.get("buggy"), x.get("fix"))
        if pair:
            bash1.append(pair)

    # 2. Safety / permissions
    d2 = load_dataset("maharshipandya/sudo-su-safety", split="train")
    bash2 = []
    for x in d2:
        pair = normalize_pair(x.get("input"), x.get("output"))
        if pair:
            bash2.append(pair)

    combined = bash1 + bash2
    random.shuffle(combined)
    return combined[:limit]


def load_critical_thinking_datasets(limit):
    print("Carregando datasets de pensamento crítico...")

    # 1. NoHallucinations
    d1 = load_dataset("RAJ9191/NoHallucinations", split="train")
    ct1 = []
    for x in d1:
        pair = normalize_pair(x.get("question"), x.get("answer"))
        if pair:
            ct1.append(pair)

    # 2. ArgumentMining (prompt -> argumento -> contra-argumento)
    # Este dataset não vem no formato tradicional, então adaptamos:
    d2 = load_dataset("allenai/ArgumentMining", "essays", split="train")
    ct2 = []
    for x in d2:
        claim = x.get("claim")
        evidence = x.get("evidence")
        if claim and evidence:
            pair = normalize_pair(claim, evidence)
            if pair:
                ct2.append(pair)

    combined = ct1 + ct2
    random.shuffle(combined)
    return combined[:limit]


# ============================================================
# Dataset final tokenizado
# ============================================================

def make_supervised_dataset(tokenizer, pairs, max_length=512):
    def gen():
        for ex in pairs:
            prompt = ex["prompt"]
            resp = ex["response"]
            full = f"Usuário: {prompt}\nAssistente: {resp}"
            enc = tokenizer(full, truncation=True, max_length=max_length)
            yield {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"]
            }
    return Dataset.from_generator(gen)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    # tamanho controlado p/ sua 3080
    LIMIT_PER_DOMAIN = 2000

    # ============================================================
    # Carregar datasets
    # ============================================================

    excel = load_excel_datasets(LIMIT_PER_DOMAIN)
    python = load_python_datasets(LIMIT_PER_DOMAIN)
    linux = load_linux_datasets(LIMIT_PER_DOMAIN)
    critical = load_critical_thinking_datasets(LIMIT_PER_DOMAIN)

    all_data = excel + python + linux + critical
    random.shuffle(all_data)

    print(f"TOTAL FINAL: {len(all_data)} exemplos combinados")

    # ============================================================
    # Tokenizer e dataset final
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_supervised_dataset(tokenizer, all_data, max_length=args.max_length)
    dataset = dataset.train_test_split(test_size=0.03)

    # ============================================================
    # Modelo aluno + LoRA
    # ============================================================

    print("Carregando modelo aluno...")
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )

    if args.use_lora:
        print("Aplicando LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ============================================================
    # Treinamento
    # ============================================================

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    print("Treinando...")
    trainer.train()

    print("Salvando modelo...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Finalizado com sucesso.")


if __name__ == "__main__":
    main()

