"""
Script para treinar um modelo aluno Qwen2.5-1.5B a partir de um modelo professor
Falcon3-10B-Instruct-AWQ utilizando distilação de conhecimento. O script também
opcionalmente gera uma base sintética crítica nos domínios de fórmulas de Excel,
código Python e administração de Arch Linux. As respostas geradas pelo professor
são críticas, apontando erros e contestando premissas.

Exemplo de uso:

    python train_distilled_llm_qwen_falcon.py \
        --teacher_model TheBloke/falcon3-10B-Instruct-AWQ \
        --student_model Qwen/Qwen2.5-1.5B-Instruct \
        --synthetic_data ./synthetic_critico_qwen.txt \
        --generate_synthetic \
        --samples_per_domain 500 \
        --output_dir ./qwen2_5-1_5b-distilled-critico \
        --use_lora \
        --fp16

Dependências:

  - transformers
  - torch
  - accelerate
  - bitsandbytes
  - peft (opcional, para LoRA)

Executar em um ambiente com GPU e suporte a 4-bit/8-bit, como uma
placa com pelo menos 24 GB se usar o professor de 10B integralmente.
"""

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

# Tentar importar LoRA; se não estiver disponível, habilitar flag PEFT_AVAILABLE
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# 1. Dataset simples: 1 linha = 1 exemplo
# =============================================================================
class SyntheticTextDataset(Dataset):
    """Dataset para textos sintéticos.

    Cada linha do arquivo de dados representa um exemplo. O dataset
    tokeniza cada linha e define labels iguais aos input_ids, permitindo
    treinamento de linguagem causal.
    """

    def __init__(self, tokenizer, data_path: str, max_length: int = 512):
        assert os.path.exists(data_path), f"Arquivo de dados não encontrado: {data_path}"
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            # Remover linhas vazias
            self.examples = [l.strip() for l in f if l.strip()]

        if len(self.examples) == 0:
            raise ValueError("Arquivo de dados está vazio.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Remover dimensão extra
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Definir labels como input_ids para treinamento de linguagem causal
        item["labels"] = item["input_ids"].clone()
        return item


# =============================================================================
# 2. Data collator simples
# =============================================================================
@dataclass
class DataCollatorForCausalLM:
    """Data collator para modelos causais.

    Usa padding do tokenizer para gerar lotes de tamanho variável com
    preenchimento apropriado.
    """

    tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )
        return batch


# =============================================================================
# 3. Trainer com distillation
# =============================================================================
class DistillationTrainer(Trainer):
    """Trainer que combina perda de distilação (KL) com cross-entropy padrão.

    A perda total é uma combinação de:
      - alpha_mle * cross-entropy com os labels verdadeiros
      - alpha_ce * KL-divergence entre logits do aluno e do professor

    A temperatura pode ser ajustada para suavizar as distribuições.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        temperature: float = 1.0,
        alpha_ce: float = 0.5,
        alpha_mle: float = 0.5,
        **kwargs,
    ):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_mle = alpha_mle

        # Congelar parâmetros do professor
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computa a perda combinada de CE e KL para distilação."""
        labels = inputs.get("labels")

        # Output do aluno
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits

        # Output do professor (sem gradiente)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            teacher_logits = outputs_teacher.logits

        # Cross-entropy padrão com rótulos
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss_ce = loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        # Distillation via KL
        t = self.temperature
        log_probs_student = torch.nn.functional.log_softmax(student_logits / t, dim=-1)
        probs_teacher = torch.nn.functional.softmax(teacher_logits / t, dim=-1)
        loss_kl = torch.nn.functional.kl_div(
            log_probs_student,
            probs_teacher,
            reduction="batchmean",
        ) * (t * t)

        # Combinação das perdas
        loss = self.alpha_mle * loss_ce + self.alpha_ce * loss_kl

        return (loss, outputs_student) if return_outputs else loss


# =============================================================================
# 4. Geração de base sintética crítica
# =============================================================================

# Conjuntos de perguntas base para cada domínio
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
    """Constrói um prompt crítico em português para o modelo professor.

    O prompt inclui um cabeçalho instruindo a assistente a apontar erros e
    contestar premissas, especificando o domínio de interesse.
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
    """Extrai a resposta do modelo removendo o prefixo do prompt.

    Se a string 'Assistente:' estiver presente, retorna o texto após ela.
    Caso contrário, remove o prompt inicial se for prefixo e retorna o restante.
    """
    idx = full_text.find("Assistente:")
    if idx != -1:
        answer = full_text[idx + len("Assistente:"):].strip()
    else:
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()
    # Normalizar para uma linha
    answer = " ".join(answer.split())
    return answer


def generate_synthetic_dataset(
    teacher_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str,
    samples_per_domain: int = 200,
    max_new_tokens: int = 256,
):
    """Gera uma base sintética crítica utilizando o modelo professor.

    Para cada domínio (Excel, Python, Arch), gera 'samples_per_domain' exemplos.
    Cada exemplo consiste em uma pergunta do usuário (selecionada de um conjunto
    fixo de perguntas base) e uma resposta gerada criticamente pelo professor.
    O resultado é salvo em 'output_path', um exemplo por linha.
    """
    device = next(teacher_model.parameters()).device

    all_domains = [
        ("EXCEL", EXCEL_QUERIES),
        ("PYTHON", PYTHON_QUERIES),
        ("ARCH", ARCH_QUERIES),
    ]

    with open(output_path, "w", encoding="utf-8") as f_out:
        for domain_name, query_list in all_domains:
            print(f"Gerando exemplos para domínio: {domain_name}")
            for _ in range(samples_per_domain):
                user_question = random.choice(query_list)
                prompt = build_critical_system_prompt(domain_name, user_question)

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
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

                full_text = tokenizer.decode(
                    gen_outputs[0],
                    skip_special_tokens=True,
                )
                answer = extract_answer_from_generated(full_text, prompt)

                line = f"[{domain_name}] Usuário: {user_question} Assistente: {answer}"
                line = " ".join(line.split())  # normalizar espaços
                f_out.write(line + "\n")

    print(f"Base sintética gerada em: {output_path}")


# =============================================================================
# 5. Interface de linha de comando e função main
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Treino de Qwen2.5-1.5B com distilação de Falcon3-10B-Instruct-AWQ "
            "e geração de base sintética crítica."
        )
    )

    parser.add_argument("--teacher_model", type=str, required=True,
                        help="Nome ou path do modelo professor (e.g., TheBloke/falcon3-10B-Instruct-AWQ).")
    parser.add_argument("--student_model", type=str, required=True,
                        help="Nome ou path do modelo aluno (e.g., Qwen/Qwen2.5-1.5B-Instruct).")
    parser.add_argument("--synthetic_data", type=str, required=True,
                        help="Caminho para o arquivo de dados sintéticos. Será criado se --generate_synthetic.")
    parser.add_argument("--output_dir", type=str, default="./qwen2_5-1_5b-distilled-critico",
                        help="Pasta onde o modelo aluno distilado será salvo.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Comprimento máximo de tokens para cada exemplo.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Tamanho do lote por dispositivo.")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Passos de acumulação de gradiente.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Número de épocas de treinamento.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Taxa de aprendizado inicial.")
    parser.add_argument("--fp16", action="store_true",
                        help="Habilitar treino em FP16.")
    parser.add_argument("--bf16", action="store_true",
                        help="Habilitar treino em BF16.")
    parser.add_argument("--use_lora", action="store_true",
                        help="Aplicar LoRA ao modelo aluno.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperatura para distilação.")
    parser.add_argument("--alpha_ce", type=float, default=0.5,
                        help="Peso da perda de distilação (KL).")
    parser.add_argument("--alpha_mle", type=float, default=0.5,
                        help="Peso da perda de cross-entropy.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Frequência de logging durante o treino.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Frequência de salvamento do modelo.")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Proporção do dataset usada para validação.")
    parser.add_argument("--generate_synthetic", action="store_true",
                        help="Se definido, gera uma nova base sintética antes do treino.")
    parser.add_argument("--samples_per_domain", type=int, default=200,
                        help="Número de exemplos a gerar por domínio.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Carregar tokenizer do modelo aluno
    print("Carregando tokenizer do modelo aluno...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Carregar modelo professor
    print("Carregando modelo professor...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) Gerar base sintética, se solicitado
    if args.generate_synthetic:
        print("Gerando base sintética crítica...")
        generate_synthetic_dataset(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            output_path=args.synthetic_data,
            samples_per_domain=args.samples_per_domain,
            max_new_tokens=256,
        )

    # 4) Carregar dataset
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
    print(f"Total de exemplos: {n_total} | treino: {n_train} | validação: {n_eval}")

    # 5) Carregar modelo aluno com quantização 4-bit e LoRA opcional
    print("Carregando modelo aluno...")
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

    # 6) Configuração de treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["none"],
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # 7) Instanciar o trainer com distillation
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        alpha_mle=args.alpha_mle,
    )

    # 8) Treinamento
    print("Iniciando treinamento com distillation...")
    trainer.train()

    # 9) Salvar modelo
    print("Salvando modelo aluno distilado...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Treino concluído.")


if __name__ == "__main__":
    main()