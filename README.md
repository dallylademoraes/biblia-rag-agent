# ScripturaRAG

Mini-projeto para construção de um pipeline de Retrieval-Augmented Generation (RAG)
aplicado a textos bíblicos.

## Escopo inicial
- Agente especializado em textos bíblicos
- Uso de RAG para respostas fundamentadas
- Foco inicial: a definir (ex.: Novo Testamento)

## Fonte dos dados
- Tradução bíblica com licença aberta ou domínio público
- Formato: PDF ou texto estruturado

## Estrutura do projeto
- data/raw: texto original
- data/processed: texto tratado
- ingestion: ingestão e segmentação
- retrieval: recuperação de trechos

## Como rodar (Windows)
- Ative a venv do repo: `.\.venv\Scripts\Activate.ps1`
- Cheque o Ollama (conexão + embeddings): `python ingestion\ingest.py --check-ollama`
- Rodar ingestão: `python ingestion\ingest.py biblia.pdf`

## Observações
- Projeto em fase inicial
