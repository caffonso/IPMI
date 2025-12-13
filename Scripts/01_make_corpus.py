#!/usr/bin/env python3
import json, re, glob, os
from pathlib import Path

OUT_JSONL = "corpus_canon.jsonl"

ART_RE = re.compile(r"\bArt\.?\s*\d+[A-Za-z]?\b", re.IGNORECASE)
PAR_RE = re.compile(r"§\s*\d+º?")
INC_RE = re.compile(r"\binciso\s+[IVXLC]+\b|\b[IVXLC]+\b(?=[\)\.\s])", re.IGNORECASE)

def extract_refs(text: str):
    artigos = sorted(set(m.group(0).strip() for m in ART_RE.finditer(text)))
    paragrafos = sorted(set(m.group(0).strip() for m in PAR_RE.finditer(text)))
    incisos = sorted(set(m.group(0).strip() for m in INC_RE.finditer(text)))
    return {"artigos": artigos[:15], "paragrafos": paragrafos[:15], "incisos": incisos[:25]}

def infer_meta_from_filename(file_path: str):
    name = Path(file_path).stem.lower().replace("_raw","")
    # heurística simples (ajuste depois se quiser)
    if "juris" in name:
        return dict(tipo="jurisprudencia", ente="desconhecido", hierarquia="jurisprudencia", doc_title=name)
    if "manual" in name or name.startswith("dp") or "direito_previdenciario" in name:
        return dict(tipo="manual", ente="geral", hierarquia="manual", doc_title=name)
    if "constituicao" in name:
        return dict(tipo="lei", ente="estado_sp", hierarquia="constituicao_estadual", doc_title="Constituição do Estado de SP")
    if "ct_sp" in name:
        return dict(tipo="lei", ente="estado_sp", hierarquia="codigo", doc_title="CT/SP")
    if "lei_" in name or "lc_" in name:
        return dict(tipo="lei", ente="estado_sp", hierarquia="lei", doc_title=name)
    return dict(tipo="desconhecido", ente="desconhecido", hierarquia="desconhecido", doc_title=name)

def get_text(obj):
    # suporta variações comuns
    for k in ("text","texto","conteudo","content","body"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def get_source(obj, fallback_source):
    for k in ("source","fonte","arquivo","file"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return fallback_source

def main():
    raw_files = sorted(glob.glob("*_raw.jsonl"))
    if not raw_files:
        print("Nenhum *_raw.jsonl encontrado no diretório atual.")
        return

    out_count = 0
    empty_text = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for rf in raw_files:
            meta = infer_meta_from_filename(rf)
            fallback_source = Path(rf).name  # se não houver "source/fonte" no JSON

            with open(rf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    text = get_text(obj)
                    source = get_source(obj, fallback_source)

                    # refs: tenta pelo texto + também aceita artigo explícito quando vier estruturado
                    refs = extract_refs(text) if text else {"artigos":[], "paragrafos":[], "incisos":[]}
                    if obj.get("artigo") and isinstance(obj["artigo"], str):
                        if obj["artigo"] not in refs["artigos"]:
                            refs["artigos"] = [obj["artigo"]] + refs["artigos"]

                    canon = {
                        "id": obj.get("id") or obj.get("numero") or f"{Path(rf).stem}:{out_count}",
                        "source": source,
                        "source_path": f"IPMI/{meta['tipo']}/{source}",
                        "doc_id": Path(source).stem.lower().replace("-","_").replace(" ","_"),
                        "doc_title": meta["doc_title"],
                        "tipo": meta["tipo"],
                        "ente": meta["ente"],
                        "hierarquia": meta["hierarquia"],
                        "refs": refs,
                        "page_start": obj.get("page_start") or obj.get("pagina_inicio"),
                        "page_end": obj.get("page_end") or obj.get("pagina_fim"),
                        "chunk_index": obj.get("chunk_index"),
                        "text": text,
                    }

                    if not text.strip():
                        empty_text += 1

                    w.write(json.dumps(canon, ensure_ascii=False) + "\n")
                    out_count += 1

    print(f"OK: {out_count} registros -> {OUT_JSONL} | text vazio: {empty_text}")

if __name__ == "__main__":
    main()
