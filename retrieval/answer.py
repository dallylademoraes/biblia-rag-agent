"""
retrieval/answer.py

Gera uma RESPOSTA final (com "persona do agente") usando:
- Retrieval híbrido (semântico + literal) em Chroma (seu banco da Bíblia em TXT)
- LLM local via Ollama (para redigir a resposta com base nos trechos)

Fluxo:
Pergunta -> Recupera versículos relevantes -> Monta prompt com persona -> LLM -> Resposta

Como rodar:
python retrieval/answer.py "Como devo perdoar segundo Jesus?"

Requisitos:
pip install langchain-ollama langchain-chroma chromadb
Ollama rodando em http://localhost:11434
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# Configurações do projeto
# =========================
CHROMA_DIR = "data/processed/chroma_db_txt"
COLLECTION_NAME = "biblia_almeida_rc_txt"

OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding: tem que bater com a ingestão
EMBED_MODEL = "nomic-embed-text:latest"

# Modelo de chat (resposta final)
# Você pode alternar entre: "llama3.2:latest" ou "gemma3:4b"
CHAT_MODEL = "llama3.2:latest"

# Quantos versículos trazer pro contexto do LLM
FINAL_TOP_N = 25

# Retrieval
SEMANTIC_TOP_K = 10
LITERAL_PER_KEYWORD_LIMIT = 30


# =========================
# Stopwords e heurísticas
# =========================
STOPWORDS_PT = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "e", "ou", "para", "por", "com", "sem",
    "que", "quem", "como", "qual", "quais", "quando", "onde",
    "porque", "porquê", "pq",
    "é", "foi", "são", "ser", "estar", "tem", "têm", "ter",
    "me", "te", "se", "nos", "vos", "lhe", "lhes",
    "devo", "segundo",
}

BOOK_NAMES = {
    "gênesis": "Gênesis",
    "êxodo": "Êxodo",
    "levítico": "Levítico",
    "números": "Números",
    "deuteronômio": "Deuteronômio",
    "josué": "Josué",
    "juízes": "Juízes",
    "rute": "Rute",
    "salmos": "Salmos",
    "provérbios": "Provérbios",
    "isaías": "Isaías",
    "jeremias": "Jeremias",
    "mateus": "Mateus",
    "marcos": "Marcos",
    "lucas": "Lucas",
    "joão": "João",
    "romanos": "Romanos",
    "hebreus": "Hebreus",
    "apocalipse": "Apocalipse",
}


def normalize(s: str) -> str:
    return s.strip().lower()


def verse_key(meta: Dict[str, Any]) -> str:
    return f"{meta.get('livro')}|{meta.get('capitulo')}|{meta.get('versiculo')}"


def format_ref(meta: Dict[str, Any]) -> str:
    return f"{meta.get('livro')} {meta.get('capitulo')}:{meta.get('versiculo')}"


def detect_book_filter(question: str) -> Optional[str]:
    q = normalize(question)
    for k, proper in BOOK_NAMES.items():
        if k in q:
            return proper
    return None


def prefer_testament(question: str) -> Optional[str]:
    q = normalize(question)
    if any(x in q for x in ["jesus", "cristo", "evangelho", "paulo", "apóstolo"]):
        return "NT"
    return None


def extract_keywords(question: str) -> List[str]:
    """
    Keywords para busca literal.
    - aceita tokens com tamanho >= 2 (pega "fé")
    - remove stopwords
    - remove duplicados
    """
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+", " ", question.lower()).strip()
    tokens = [t for t in cleaned.split() if len(t) >= 2 and t not in STOPWORDS_PT]

    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ==========================
# Retrieval (semântico + literal)
# ==========================
def semantic_search(
    vs: Chroma,
    question: str,
    top_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    hits = vs.similarity_search_with_score(question, k=top_k, filter=where or None)
    return [(doc.page_content, doc.metadata, float(score)) for doc, score in hits]


def literal_search_ranked(
    vs: Chroma,
    keywords: List[str],
    per_kw_limit: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], int]]:
    """
    Busca literal por $contains em cada keyword e re-ranqueia pela quantidade
    de keywords presentes no texto do versículo.
    """
    if not keywords:
        return []

    keywords = keywords[:6]  # evita custo alto

    pool: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for kw in keywords:
        variants = {kw, kw.capitalize(), kw.upper()}
        for vkw in variants:
            r = vs._collection.get(
                where=where,
                where_document={"$contains": vkw},
                limit=per_kw_limit,
            )
            docs = r.get("documents", []) or []
            metas = r.get("metadatas", []) or []
            for text, meta in zip(docs, metas):
                k = verse_key(meta)
                if k not in pool:
                    pool[k] = (text, meta)

        docs = r.get("documents", []) or []
        metas = r.get("metadatas", []) or []
        for text, meta in zip(docs, metas):
            k = verse_key(meta)
            if k not in pool:
                pool[k] = (text, meta)

    ranked: List[Tuple[str, Dict[str, Any], int]] = []
    for _k, (text, meta) in pool.items():
        t = normalize(text)
        match_count = sum(1 for kw in keywords if kw in t)
        if match_count > 0:
            ranked.append((text, meta, match_count))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def merge_results(
    lit: List[Tuple[str, Dict[str, Any], int]],
    sem: List[Tuple[str, Dict[str, Any], float]],
) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Junta resultados:
    - literal primeiro (mais "exato" quando acerta)
    - semântico depois (cobre perguntas conceituais)
    Dedup por referência (livro/cap/vers).
    """
    merged: List[Tuple[str, Dict[str, Any], str]] = []
    seen = set()

    def add(text: str, meta: Dict[str, Any], source: str):
        k = verse_key(meta)
        if k in seen:
            return
        seen.add(k)
        merged.append((text, meta, source))

    for text, meta, _mc in lit:
        add(text, meta, "literal")

    for text, meta, _score in sem:
        add(text, meta, "semantic")

    return merged


def retrieve_verses(question: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Função principal de retrieval: devolve uma lista de (texto, meta, fonte).
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    book = detect_book_filter(question)
    pref = prefer_testament(question)

    where: Dict[str, Any] = {}
    if book:
        where["livro"] = book
    # Se a pergunta for explicitamente sobre Jesus/Cristo, filtra de verdade pro NT
    if pref == "NT":
        where["testamento"] = "NT"

    keywords = extract_keywords(question)

    is_bio = question.lower().strip().startswith(("quem é", "quem foi"))

    if is_bio:
        # Pergunta biográfica → força literal, evita semântico
        lit = literal_search_ranked(
            vs,
            keywords,
            per_kw_limit=80,
            where=where or None
        )
        sem = []
    else:
        lit = literal_search_ranked(
            vs,
            keywords,
            LITERAL_PER_KEYWORD_LIMIT,
            where=where or None
        )
        sem = semantic_search(
            vs,
            question,
            SEMANTIC_TOP_K,
            where=where or None
        )

    merged = merge_results(lit, sem)
    return merged[:FINAL_TOP_N]


# ==========================
# Prompt do agente (persona)
# ==========================
def build_context_block(verses: List[Tuple[str, Dict[str, Any], str]]) -> str:
    """
    Formata os versículos recuperados para serem colocados no prompt.
    """
    lines = []
    for text, meta, source in verses:
        ref = format_ref(meta)
        lines.append(f"- [{ref}] ({source}) {text}")
    return "\n".join(lines)


def build_messages(question: str, context_block: str) -> List:
    """
    Mensagens para o LLM:
    - SystemMessage define persona + regras
    - HumanMessage traz pergunta + contexto
    """
    system = SystemMessage(
        content=(
            "Você é um assistente especialista em Bíblia (Almeida Revista e Corrigida). "
            "RESPONDA EXCLUSIVAMENTE com base no CONTEXTO fornecido. "
            "É PROIBIDO usar conhecimento externo, memória prévia ou inferências fora do texto. "
            "Se a informação não estiver explicitamente presente no CONTEXTO, "
            "responda: 'O contexto fornecido não contém informação suficiente para responder.' "
            "Toda afirmação deve citar ao menos um versículo do CONTEXTO. "
            "Nunca cite versículos que não apareçam no CONTEXTO."
        )
    )

    user = HumanMessage(
        content=(
            f"PERGUNTA:\n{question}\n\n"
            f"CONTEXTO (versículos recuperados):\n{context_block}\n\n"
            "INSTRUÇÕES DE RESPOSTA:\n"
            "- Comece com uma definição objetiva.\n"
            "- Use apenas informações presentes no CONTEXTO.\n"
            "- Cite os versículos logo após cada afirmação.\n"
            "- Não mencione personagens, eventos ou livros fora do CONTEXTO.\n"
        )
    )

    return [system, user]


# ==========================
# Geração da resposta
# ==========================
def answer(question: str) -> str:
    verses = retrieve_verses(question)
    # Garante ordem narrativa correta (livro > capítulo > versículo)
    verses.sort(
        key=lambda x: (
            x[1].get("livro"),
            x[1].get("capitulo"),
            x[1].get("versiculo"),
        )
    )
    # Se pergunta é "Quem é/foi X", exige que o nome apareça no contexto
    q = question.lower()
    is_bio = q.strip().startswith(("quem é", "quem foi"))
    if is_bio:
        # pega um "nome candidato" simples (primeira palavra após "quem é/foi")
        parts = re.sub(r"[^\wÀ-ÖØ-öø-ÿ\s]", " ", q).split()
        target = None
        if len(parts) >= 3 and parts[0] == "quem" and parts[1] in ("é", "foi"):
            target = parts[2]

        if target:
            has_name = any(target.lower() in (text or "").lower() for text, _meta, _src in verses)
            if not has_name:
                return "Não encontrei no contexto recuperado versículos que mencionem explicitamente esse nome."

    if not verses:
        return "Não encontrei versículos relevantes para essa pergunta. Tente reformular com mais contexto."

    context_block = build_context_block(verses)
    messages = build_messages(question, context_block)

    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    result = llm.invoke(messages)

    # result pode ser AIMessage; .content contém o texto
    return result.content


def main() -> int:
    if len(sys.argv) < 2:
        print('Uso: python retrieval/answer.py "sua pergunta"')
        return 1

    question = sys.argv[1]
    print(answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
