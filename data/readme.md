###  Schema canônico recomendado para o RAG

Crie um arquivo 01_make_corpus.py e rode em cima dos seus *_raw.jsonl.

Ele:

lê todos os JSONL raw

normaliza campos

infere tipo/ente/hierarquia a partir do nome do arquivo

extrai referências simples por regex (Art., §, inciso)

escreve corpus_canon.jsonl
