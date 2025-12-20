## 1) Princípio de arquitetura: “LLM para extrair e explicar; motor determinístico para calcular e decidir”

Para IPMI, o maior ganho vem de separar **extração** (incerta) de **cálculo/decisão** (deve ser determinística e auditável).

* **LLM / IA**: leitura de PDF, identificação de campos, normalização, detecção de inconsistências, geração de “memória de cálculo” (explicação).
* **Regra determinística (rules engine)**: cálculo de tempo, base de contribuição, média, proventos, DIB/DIP, regras de transição, elegibilidade.
  Isso reduz risco jurídico e melhora repetibilidade.

---

## 2) Pipeline recomendado (de ponta a ponta)

### Camada A — Ingestão e triagem (rápido de implementar)

**Objetivo:** receber PDFs e classificá-los automaticamente por tipo e por origem.

1. **Upload + armazenamento**

   * Pasta/objeto por servidor: `matricula/ano/documento.pdf`
   * Armazenamento local ou MinIO (S3 local), com hash para evitar duplicidade.

2. **Classificação automática de documentos (LLM leve + heurística)**
   Tipos comuns (exemplos):

   * Carteira de trabalho (CTPS)
   * CNIS/INSS (quando existir)
   * “Certidão/Certificado de Tempo de Serviço” da Prefeitura
   * “Declaração” mensal IPMI (base previdenciária + contribuição)
   * Holerites/“Discriminação de salários e contribuições mês/ano”

**Saída:** cada PDF vira um “lote” com: tipo, período coberto, confiabilidade, e páginas relevantes.

---

### Camada B — Extração estruturada (o coração do projeto)

Você descreveu campos críticos. O ideal é transformar tudo em tabelas padronizadas.

#### B1) Extração com layout + OCR quando necessário

* Se PDF tem texto selecionável: extrair via `PyMuPDF`/`pdfplumber`.
* Se é imagem/scan: OCR (Tesseract/PaddleOCR/DocTR) + detecção de tabelas.

**Estratégia prática:**

* Primeiro tenta texto nativo; se “densidade de caracteres” baixa, cai para OCR.
* Para tabelas (mês/ano/valores), usar extração tabular (Camelot/Tabula) quando funcionar, e **fallback** para LLM “table reconstruction”.

#### B2) “Schema-first extraction” (LLM preenchendo um JSON rígido)

Defina *schemas* por tipo de documento. Exemplo de saídas:

1. **Vínculos (CTPS / empresas / Prefeitura)**
   `employment_periods`:

* empregador
* tipo (privado / prefeitura)
* regime (INSS / RPPS/IPMI)
* cargo (se houver)
* dt_inicio, dt_fim
* observações

2. **Contribuições mensais**
   `monthly_contributions` (chave = competência AAAA-MM):

* base_previdenciaria
* contribuicao_servidor
* contribuicao_patronal (se aplicável)
* fonte (holerite/declaração)
* confiança

3. **Certidão de tempo de serviço (Prefeitura)**
   `service_time_certificate_rows`:

* ano (ou período)
* tempo_bruto
* faltas
* licencas
* suspensao
* tempo_liquido (ou campos para calcular)
* total_acumulado (se existir)

**Importante:** faça o LLM sempre devolver:

* `valor_extraido`
* `evidencia` (trecho + página)
* `confidence` (0–1)
* `normalizacao` (ex.: “R$ 1.234,56” → 1234.56)

Isso permite auditoria e revisão humana.

---

### Camada C — Reconciliação e validação (onde você ganha confiabilidade)

**Problema real:** holerite e declaração podem divergir; certidão anual pode conflitar com somatório mensal; CTPS pode divergir de CNIS.

Implemente um “agente de conciliação” (ou módulo) com regras simples:

* **Regra de prioridade de fonte** (ajustável pelo IPMI):

  1. Declaração oficial IPMI
  2. Certidão de tempo de serviço homologada
  3. Holerite
  4. CTPS (para vínculo, não para contribuição)
* **Detecção de buracos (gaps)**: meses faltantes dentro de um vínculo.
* **Detecção de sobreposição**: dois vínculos no mesmo mês.
* **Saldos impossíveis**: contribuição > base * alíquota máxima esperada (sinaliza).
* **Unificação por competência**: sempre consolidar por AAAA-MM.

**Saída:** um “dossiê consolidado” com pendências e justificativa.

---

### Camada D — Motor de regras para cálculo e decisão (determinístico)

Aqui você implementa “o que de fato decide”:

1. **Tempo de contribuição / serviço**

* total por regime (INSS vs RPPS/IPMI)
* tempo no cargo / carreira / serviço público (se exigido)
* abatimentos (faltas/licenças/suspensões conforme regramento local)
* conversões (se houver hipóteses legais)

2. **Elegibilidade e data de início (DIB / DIP)**

* checa requisitos (idade, tempo, regras de transição, etc.)
* define:

  * **data de implementação dos requisitos**
  * **DIB** (data de início do benefício) conforme regra aplicável e requerimento
  * **DIP** (data de início do pagamento), quando houver distinção

3. **Cálculo do valor (proventos)**
   Depende totalmente do estatuto/lei local e regras pós-reforma. Por isso:

* codifique fórmulas em módulo separado
* versão as regras (ex.: “LC municipal X/ano”, “Emenda Y”)
* gere memória de cálculo linha a linha (mês a mês quando necessário)

**Saída final:** resultado numérico + trilha de cálculo.

---

### Camada E — Geração de relatório e explicação (LLM de “redação controlada”)

O LLM entra no fim para produzir um documento claro para:

* servidor (linguagem simples)
* analista (detalhamento com tabelas)
* auditoria/controle interno (referências e evidências)

Formato recomendado (padronizado):

1. Identificação do segurado e fontes analisadas
2. Linha do tempo de vínculos
3. Quadro mensal consolidado (competência, base, contribuição, fonte)
4. Totalizações (por regime e total)
5. Elegibilidade (requisitos atendidos e em que data)
6. Cálculo do valor (fórmula e memória)
7. Pendências / inconsistências (se houver)
8. Conclusão (DIB/DIP e valor estimado)

---

## 3) Onde usar GPU (de modo racional)

Você tem GPU e notebooks. Uma divisão eficiente:

* **GPU**: OCR mais pesado (PaddleOCR/DocTR), modelos LLM locais (ex.: 7B–14B quantizados) para extração e redação, reranking/embeddings se necessário.
* **Notebooks/CPU**: FastAPI, rules engine, banco de dados, interface, filas.

---

## 4) Stack de implementação (realista para você)

Sem “engenharia infinita”, mas robusto:

* **Backend**: Python + FastAPI
* **Filas** (para processar PDFs em lote): Celery + Redis (ou RQ + Redis)
* **Banco**:

  * Postgres (recomendado) para dados estruturados
  * Armazenamento de arquivos: MinIO ou filesystem com metadados no Postgres
* **Extração PDF**: PyMuPDF + pdfplumber
* **OCR**: PaddleOCR (bom custo/benefício) ou Tesseract (mais simples)
* **Validação/normalização**: Pydantic (schemas)
* **LLM**:

  * Modelo local quantizado para extração e sumarização
  * Prompt “schema-first” + validação Pydantic (reexecuta se inválido)
* **UI interna**:

  * Comece com Streamlit (MVP rápido)
  * Migre para React/Next se necessário

---

## 5) “Agentes” que fazem sentido (sem complicar)

Você pode implementar como módulos/serviços, não precisa de um framework complexo de agentes.

1. **Agente de Triagem**: classifica documento + encontra páginas relevantes.
2. **Agente Extrator**: gera JSON por tipo (com evidências).
3. **Agente Conciliador**: une, prioriza fontes, marca conflitos.
4. **Agente Calculador** (determinístico): aplica regras, calcula.
5. **Agente Redator**: gera relatório final controlado.

---

## 6) Otimizações de processo além do LLM (alto impacto, baixo custo)

1. **Checklist de documentos** automatizado
   Após ingestão, o sistema já retorna: “faltam competências X–Y”, “falta CTPS página Z”, etc.

2. **Modelo único de competência**
   Tudo vira AAAA-MM. Isso sozinho reduz erros e tempo de análise.

3. **Tela de revisão humana por exceção**
   Analista só revisa itens com:

   * baixa confiança
   * conflito entre fontes
   * buracos/overlaps
     Resultado: você reduz trabalho manual em 60–80% em casos “limpos”.

4. **Catálogo de regras versionado**
   Cada alteração legal vira uma nova versão do motor. Isso evita “decisão mutante”.

5. **Trilha de auditoria**
   Guarde: PDF original, extrações, evidências, versão das regras, resultado final.

---

## 7) Plano de execução em 4 sprints (objetivo: entregar valor rápido)

### Sprint 1 — MVP de ingestão + extração básica (1–2 semanas)

* Upload + classificação (3–5 tipos)
* Extração de tabelas “mês/ano/valor” com 1 ou 2 layouts
* Armazenar JSON + evidências por página

### Sprint 2 — Consolidação por competência + detecção de inconsistências (1–2 semanas)

* Unificar tudo em `monthly_contributions`
* Regras de conciliação e flags
* Tela Streamlit para revisar exceções

### Sprint 3 — Motor determinístico de tempo + elegibilidade (2–4 semanas)

* Cálculo de tempo (por regime) e data de implementação
* Primeira versão de DIB/DIP conforme regra do município
* Relatório preliminar

### Sprint 4 — Cálculo de valor + relatório completo + hardening (3–6 semanas)

* Implementar fórmula(s) municipal(is) de proventos
* Memória de cálculo
* Logs, versionamento de regras, permissões, LGPD

---

## 8) Riscos comuns e como mitigar

* **PDF muito heterogêneo**: use “biblioteca de layouts” por tipo de documento (templates) e fallback LLM.
* **Erros silenciosos do LLM**: schema rígido + validação + evidência obrigatória + revisão por exceção.
* **Mudança legal/regra**: rules engine versionado, jamais “hardcode sem versão”.
* **Confiabilidade jurídica**: LLM não decide; só extrai e redige; decisão é determinística e auditável.

---

## 9) Entregáveis concretos que recomendo você construir

1. **Banco de dados** com:

   * pessoas/servidores
   * documentos e metadados
   * tabelas mensais consolidada
   * vínculos
   * certidões anuais
   * resultados (tempo, elegibilidade, DIB/DIP, valor)
   * pendências

2. **Relatório padrão IPMI** (PDF/HTML) com evidências por página.

3. **Painel de triagem**:

   * “casos prontos para cálculo”
   * “casos com pendência documental”
   * “casos com conflito de fontes”

---

## 10) Próximo passo (sem depender de novas perguntas)

Se você começar amanhã, o melhor “primeiro alvo” é **a tabela de contribuições por competência (AAAA-MM)**, porque:

* é base para quase todo cálculo
* permite medir ganho rapidamente
* destrava o restante do pipeline

Se você quiser, eu já posso te entregar na sequência:

* um **modelo de schema Pydantic** para `monthly_contributions`, `employment_periods`, `service_time_certificate_rows`;
* um **esqueleto FastAPI + Celery + Redis** com endpoints de upload e um worker de extração;
* prompts “schema-first” com política de evidência e re-tentativa automática quando o JSON vier inválido.

Basta você me dizer: você vai rodar isso em Windows ou Linux no servidor com GPU, e qual OCR você prefere começar (Tesseract pela simplicidade, ou PaddleOCR pela qualidade)?
