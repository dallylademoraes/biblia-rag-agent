# 📖 ScripturaRAG

ScripturaRAG é um mini-projeto que explora o uso de **Retrieval-Augmented Generation (RAG)** para a construção de um agente de IA especializado em textos bíblicos, com respostas fundamentadas diretamente nas Escrituras.

O objetivo do projeto é demonstrar, de forma prática e estruturada, como pipelines RAG podem ser aplicados a um domínio de conhecimento específico, priorizando **fidelidade textual**, **referências explícitas** e **redução de alucinações**.

---

## 🎯 Objetivos do Projeto

- Construir um agente de IA especializado em textos bíblicos
- Implementar um pipeline RAG para recuperação de trechos relevantes
- Garantir respostas fundamentadas com referências a livros, capítulos e versículos
- Explorar boas práticas de ingestão, chunking e recuperação de documentos
- Avaliar a confiabilidade das respostas em um domínio textual estruturado

---

## 🧠 Abordagem

O projeto utiliza o paradigma de **Retrieval-Augmented Generation (RAG)**, no qual:

1. Textos bíblicos são ingeridos e segmentados respeitando sua estrutura (livro, capítulo e versículo)
2. Trechos relevantes são recuperados a partir de uma base vetorial
3. As respostas do agente são geradas com base exclusivamente nos textos recuperados
4. Cada resposta deve apresentar referências explícitas às passagens utilizadas

O **LangChain** é utilizado como framework principal para orquestração do pipeline.

---

## 🗂️ Estrutura do Projeto

```text
scriptura-rag/
├── data/
│   ├── raw/          # Arquivos originais (ex.: PDF da Bíblia)
│   └── processed/    # Textos processados e estruturados
│
├── ingestion/        # Etapa de ingestão e segmentação dos textos
├── retrieval/        # Estratégias de recuperação de informação
├── agent/            # Lógica do agente e regras de resposta
├── evaluation/       # Perguntas de teste e critérios de avaliação
│
└── README.md
