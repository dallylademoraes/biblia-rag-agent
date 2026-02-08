# Agente Bíblia RAG

Um agente RAG (Retrieval‑Augmented Generation) com UI estilo ChatGPT que responde perguntas bíblicas **usando e citando versículos recuperados da própria Bíblia** como contexto.

Este projeto foi feito para demonstrar, de ponta a ponta:
- pipeline de ingestão (TXT → documentos com metadados)
- embeddings e vector store local (Chroma)
- recuperação semântica (retrieval)
- geração com “guardrails” (o modelo responde só com base no contexto)
- UI web simples com API local (`POST /api/answer`)

Criado por **Dallyla de Moraes**.

## Destaques (para avaliação)
- **RAG com citação de fonte**: a resposta referencia livro/capítulo/versículo do contexto recuperado.
- **Multilíngue/Português**: embeddings Cohere `embed-multilingual-v3.0`.
- **Vector store local**: ChromaDB persistido em `data/processed/chroma_db_txt`.
- **Configuração simples**: chaves via `.env` e modelo Groq via `GROQ_MODEL`.
- **Resiliência a deprecações**: `retrieval/answer.py` tenta fallback se um modelo Groq estiver descontinuado.
- **UI pronta para demo**: indicador “Agente pronto” quando o ambiente está OK + cache em memória.

## Stack
- Python + LangChain
- ChromaDB (vector store local)
- Cohere Embeddings (`embed-multilingual-v3.0`)
- Groq Chat (modelo configurável via `GROQ_MODEL`)
- UI: HTML/CSS/JS (vanilla) + servidor HTTP stdlib (`ui/server.py`)

## Arquitetura (alto nível)
1. **Ingestão** (`ingestion/ingest.py`)
   - Lê `data/raw/biblia.txt`
   - Segmenta por versículo e adiciona metadados (`livro`, `capitulo`, `versiculo`, `testamento`)
   - Gera embeddings e grava no Chroma (`data/processed/chroma_db_txt`)
2. **Retrieval + Answer** (`retrieval/answer.py`)
   - Busca semântica no Chroma (top‑k versículos)
   - Monta um contexto com referências e pergunta
   - O LLM responde **somente com base no contexto**
3. **UI/API** (`ui/server.py` + `ui/static/*`)
   - `POST /api/answer` chama `retrieval.answer.answer()`
   - `GET /api/health` valida chaves e existência do DB
   - Frontend conversa com a API e exibe o estado “Agente pronto”

## Como rodar (demo local)
### Pré‑requisitos
- Python 3.10+
- Chaves: Cohere + Groq

### 1) Instalar dependências
Opcional (se você não tiver venv):
- Windows (PowerShell): `python -m venv .venv` e `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `python3 -m venv .venv` e `source .venv/bin/activate`

Depois, instale:
- `pip install -r requirements.txt`

### 2) Configurar `.env`
Crie um arquivo `.env` na raiz do projeto:

```env
COHERE_API_KEY=...
GROQ_API_KEY=...
# opcional: override do modelo
GROQ_MODEL=llama-3.3-70b-versatile

# recomendado para demo/deploy controlado (evita gastar a cota grátis):
DEMO_MODE=1
# opcional: exige uma chave no front (compartilhe só com recrutadores)
DEMO_TOKEN=troque_isto_por_algo_aleatorio
# limites (ajuste conforme sua cota)
DEMO_MAX_REQ_PER_IP_PER_DAY=15
DEMO_MAX_REQ_TOTAL_PER_DAY=80
DEMO_COOLDOWN_S=3
DEMO_MAX_CHARS=280
```

### 3) Rodar a ingestão (1x)
- `python ingestion/ingest.py`

### 4) Rodar a UI
- `python ui/server.py`
- Abra `http://127.0.0.1:8000`

**Cache (opcional):**
- TTL: `python ui/server.py --cache-ttl 600`
- Máximo: `python ui/server.py --cache-max 200`

## Usar via CLI (sem UI)
- `python retrieval/answer.py "Quem foi Moisés?"`

## Exemplos de perguntas (para demo)
- `Quem foi Moisés?`
- `O que a Bíblia diz sobre amor?`
- `Onde Jesus nasceu?`
- `Juízes 16 fala sobre o quê?`

## Estrutura do repo
- `data/raw/`: fonte (`biblia.txt`)
- `data/processed/`: saída da ingestão (Chroma DB)
- `ingestion/`: pipeline de ingestão
- `retrieval/`: retrieval + geração da resposta (RAG)
- `ui/`: servidor e frontend

## Troubleshooting rápido
- Banco não existe: rode `python ingestion/ingest.py`
- Falta `COHERE_API_KEY`: configure no `.env`
- Falta `GROQ_API_KEY`: configure no `.env`

## Observações
- As respostas são limitadas ao conteúdo do arquivo em `data/raw/biblia.txt` e ao que for recuperado no contexto (comportamento intencional de RAG).
- Não versionar chaves: mantenha `.env` fora do git.
