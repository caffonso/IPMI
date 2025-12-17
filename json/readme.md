## Modelo
export EMBED_MODEL_ID=intfloat/multilingual-e5-base
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

## Estrutura
App -> LLM
rag_engine -> constroi o rag
faiss - > estrutura o jsonl

## Semantica

Sugestões
1) build_faiss: chunking orientado a “unidades normativas” (não por tamanho fixo)
2) build_faiss: embeddings “instruct” e normalização de vetores
3) build_faiss: multi-index (ou pelo menos campos para “priorização”)
4) rag_engine: retrieval híbrido (BM25 + vetorial) com “fusion”
5) rag_engine: query rewriting “jurídico” e decomposição do caso
6) rag_engine: reranking (obrigatório para “precisão jurídica”)
7) rag_engine: montagem de contexto com “cotas” por tipo de fonte
8) rag_engine: prompt de resposta com “checklist” e exigência de citação

Para “análise de casos”, o prompt deve forçar:
qual benefício foi pedido
requisitos legais (idade, carência, qualidade de segurado, etc.)
aplicação ao caso
conclusão (deferir/indeferir e por quê)
alternativas (BPC, complementação, indenização/retroativo se existisse, etc.)
citações: cada afirmação normativa importante com chunk-id/página

Avaliação

Causa combinada:

Chunking fraco (competência misturada com benefícios)
Retrieval só semântico
context_has_legal_support permissivo
Prompt sem checklist de requisitos

rag_engine.py 

1.melhorias estruturais (prioridade alta)
1.1. Retrieval atual é apenas semântico (FAISS puro)
1.2. Falta “peso jurídico” nos metadados
1.3. context_has_legal_support é fraco semanticamente

build_faiss 

2. melhorias que você deve aplicar (mesmo não enviado aqui)
2.1. Chunking atual provavelmente é linear
2.2. Falta “index_meta.json”

app.py 

3.melhorias no fluxo de decisão (alto impacto)
3.1. Prompt não força checklist jurídico
Antes de concluir, verifique explicitamente:
- Benefício requerido
- Requisitos legais
- Quais requisitos foram cumpridos
- Quais não foram cumpridos
3.2. Geração sem “query decomposition”
“aposentadoria por idade urbana requisitos”
“carência RGPS 180 contribuições”
“Lei 8.213 art. 25”
“alternativa BPC idoso 65”

ask.py


## PROMPT DE GERAÇÃO – RAG PREVIDENCIÁRIO (SPPREV)


---

Você é um assistente jurídico especializado em direito previdenciário do Regime Próprio do Estado de São Paulo (RPPS/SP).

Use apenas os trechos recuperados pelo RAG.
Não copie artigos de lei literalmente.

Ao responder um caso concreto, siga exatamente esta estrutura:
1. Tema jurídico
2. Requisitos aplicáveis (resumo)
(idade, tempo de contribuição, tempo em atividade específica, se houver)
3. Aplicação ao caso
(compare os dados do servidor com os requisitos)
4. Conclusão direta
(diga claramente se os requisitos foram ou não cumpridos)
Regras importantes:
Não tratar de assuntos que não foram perguntados.
Diferenciar benefício de contribuição.
Se o texto recuperado não for pertinente, diga que não há base suficiente.
Estilo:
Frases curtas, linguagem técnica simples, sem citações extensas.

Exemplo de pergunta.

Carla é investigador de polícia civil do Estado de São Paulo, com 55 anos de idade e 30 anos de contribuição, 
sendo 25 deles em atividade policial. Requer aposentadoria com fundamento em regras especiais para segurança pública. 
A SPPREV questiona se o tempo mínimo em atividade policial foi efetivamente cumprido e se o servidor atende aos novos requisitos pós-reforma

