# Rag na Bíblia

Chat web (estilo ChatGPT) com RAG para responder perguntas bíblicas usando trechos recuperados da própria Bíblia como contexto.

## Fonte dos dados
- Tradução bíblica com licença aberta ou domínio público
- Formato: TXT estruturado (`data/raw/biblia.txt`)

## Proposta (como funciona)
- Ingestão: parseia o TXT em versículos com metadados (`livro`, `capitulo`, `versiculo`, `testamento`) e salva no Chroma.
- Retrieval: para cada pergunta, recupera os versículos mais relevantes por similaridade.
- Geração: o LLM responde **usando apenas o contexto recuperado**; se a resposta não estiver no contexto, ele deve dizer que não sabe.
- UI: chat no navegador consumindo `POST /api/answer`.

## Estrutura do projeto
- data/raw: texto original
- data/processed: texto tratado
- ingestion: ingestão e segmentação
- retrieval: recuperação de trechos
- ui: front estilo ChatGPT

## Como rodar (Windows)
- Ative a venv do repo: `.\.venv\Scripts\Activate.ps1`
- Configure a chave do Google (Gemini/Embeddings): crie `.env` com `GOOGLE_API_KEY=...`
- Rodar ingestão (gera `data/processed/chroma_db_txt`): `python ingestion\ingest.py`
- Rodar chat (UI): `python ui\server.py` (abre `http://127.0.0.1:8000`)
  - Cache em memória (para perguntas repetidas ficarem instantâneas):
    - TTL padrão: 600s (10 min) — configure com `python ui\server.py --cache-ttl 600`
    - Tamanho máximo: 200 itens — configure com `python ui\server.py --cache-max 200`

## Modelos (atual)
- Embeddings: Google `models/text-embedding-004` (definido em `ingestion/ingest.py`)
- LLM: `gemini-2.5-flash` (definido em `retrieval/answer.py`)

## Migração: Ollama → Gemini (latência e trade-offs)
Eu comecei com Ollama local, mas (sem GPU) a latência do LLM ficou alta (respostas demorando muito em CPU). A troca para Gemini resolveu o “peso” do modelo local, porém trouxe novos pontos de atenção:
- Latência variável de rede (picos por conexão/rota/servidor) e dependência de internet.
- Rate limits/cotas e necessidade de chave/API (e custo por uso, dependendo do plano).
- Necessidade de manter o contexto curto (menos versículos) para reduzir tempo de ida/volta e tokens.

Para mitigar latência no dia a dia:
- Use o cache do `ui/server.py` e mantenha o `k` do retrieval baixo.
- Evite prompts longos e perguntas muito abertas (que puxam mais contexto).
