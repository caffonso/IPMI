#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------
# CONFIGURAÃ‡ÃƒO DO MODELO HF
# --------------------------

MODEL_ID = "ehristoforu/Falcon3-MoE-2x7B-Insruct"

print(">> Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(">> Carregando modelo (isso pode demorar na primeira vez)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # usa GPU/CPU automaticamente
)

app = FastAPI(title="Falcon3 Web Chat")


# --------------------------
# MODELOS DE REQUISIÃ‡ÃƒO
# --------------------------

class ChatRequest(BaseModel):
    message: str
    model: str | None = None  # mantido sÃ³ por compatibilidade, nÃ£o usamos


# --------------------------
# PÃGINA HTML SIMPLES
# --------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Chat com Falcon3</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 20px auto; }
    #log { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: auto; white-space: pre-wrap; }
    textarea { width: 100%; height: 80px; }
    button { padding: 8px 16px; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Chat com Falcon3-MoE-2x7B-Insruct</h1>
  <div id="log"></div>
  <textarea id="input" placeholder="Digite sua pergunta..."></textarea>
  <br>
  <button onclick="sendMessage()">Enviar</button>

  <script>
    const log = document.getElementById('log');
    const input = document.getElementById('input');

    function append(role, text) {
      log.textContent += role + ": " + text + "\\n\\n";
      log.scrollTop = log.scrollHeight;
    }

    async function sendMessage() {
      const msg = input.value.trim();
      if (!msg) return;
      append("VocÃª", msg);
      input.value = "";

      try {
        const resp = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: msg })
        });

        let data;
        try {
          data = await resp.json();
        } catch (e) {
          append("Erro", "Resposta invÃ¡lida do servidor.");
          return;
        }

        if (data.reply) {
          append("Modelo", data.reply);
        } else if (data.error) {
          append("Erro", "Backend: " + data.error);
        } else {
          append("Erro", "Servidor respondeu sem campo 'reply'.");
        }

      } catch (e) {
        append("Erro", "Falha ao chamar o servidor: " + e);
      }
    }
  </script>
</body>
</html>
"""


# --------------------------
# ROTAS
# --------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
  return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Recebe uma mensagem do usuÃ¡rio e chama o modelo Falcon3 via Transformers.
    """
    user_text = (req.message or "").strip()
    if not user_text:
        return JSONResponse({"error": "Mensagem vazia."}, status_code=400)

    print(">> Pergunta:", user_text, flush=True)

    try:
        # Prompt simples; depois podemos sofisticar com template de chat.
        prompt = user_text

        inputs = tokenizer(prompt, return_tensors="pt")
        # Garante que os tensores estÃ£o no mesmo device do modelo
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Corta a parte do input e pega sÃ³ a continuaÃ§Ã£o
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print("<< Resposta gerada com", len(reply), "caracteres", flush=True)
        return JSONResponse({"reply": reply})

    except Exception as e:
        print("!! ERRO no modelo:", repr(e), flush=True)
        return JSONResponse({"error": str(e)}, status_code=500)
