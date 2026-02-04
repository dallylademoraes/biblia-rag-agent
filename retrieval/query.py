# retrieval/query.py
"""
Retrieval híbrido (GERAL) para Bíblia em TXT.

O que este script faz:
- Suporta perguntas "quem", "como", "por que", "o que é", etc.
- Mistura 2 estratégias:
  1) Busca SEMÂNTICA (embeddings): boa para perguntas conceituais ("O que é fé?")
  2) Busca LITERAL (keywords): boa para nomes próprios/termos exatos ("Dalila", "Sansão", "Juízes 16")

Principais melhorias vs. um híbrido ingênuo:
- Keywords: filtramos palavras fracas ("devo", "segundo", "perdeu" etc.)
- Literal: re-ranqueamos por QUANTAS keywords o versículo contém (não só "contém uma")
- Filtros por metadados: se a pergunta menciona um livro (Juízes) ou Jesus (preferir NT), usamos isso
- Deduplicação: evita repetir o mesmo versículo várias vezes

Observação importante:
- EMBED_MODEL precisa ser o mesmo modelo usado na ingestão do TXT (dimensão do embedding).
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# =========================
# Configurações do projeto
# =========================
CHROMA_DIR = "data/processed/chroma_db_txt"
COLLECTION_NAME = "biblia_almeida_rc_txt"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"  # precisa bater com a ingestão

# Quantos candidatos pegar em cada modo
SEMANTIC_TOP_K = 10
LITERAL_PER_KEYWORD_LIMIT = 30  # por keyword (depois a gente corta/ranqueia)
FINAL_TOP_N = 10


# =========================
# Stopwords e heurísticas
# =========================
# Stopwords bem agressivas para evitar ruído no literal.
# (Pode ajustar depois, mas assim já melhora MUITO.)
STOPWORDS_PT = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "e", "ou", "para", "por", "com", "sem",
    "que", "quem", "como", "qual", "quais", "quando", "onde",
    "porque", "porquê", "pq",
    "é", "foi", "são", "ser", "estar", "tem", "têm", "ter",
    "me", "te", "se", "nos", "vos", "lhe", "lhes",
    # Palavras que estragam o literal (você viu na prática)
    "devo", "segundo", "perdeu", "força", "coisa", "isso", "isto", "sobre",
}

# Um mini conjunto de livros para filtro quando a pergunta mencionar explicitamente.
# (Você pode ir expandindo com o tempo, mas isso já cobre bastante.)
BOOK_NAMES = {
    "gênesis": "Gênesis",
    "êxodo": "Êxodo",
    "levítico": "Levítico",
    "números": "Números",
    "deuteronômio": "Deuteronômio",
    "josué": "Josué",
    "juízes": "Juízes",
    "rute": "Rute",
    "i samuel": "I Samuel",
    "ii samuel": "II Samuel",
    "i reis": "I Reis",
    "ii reis": "II Reis",
    "salmos": "Salmos",
    "provérbios": "Provérbios",
    "isaías": "Isaías",
    "jeremias": "Jeremias",
    "mateus": "Mateus",
    "marcos": "Marcos",
    "lucas": "Lucas",
    "joão": "João",
    "romanos": "Romanos",
    "coríntios": "Coríntios",
    "gálatas": "Gálatas",
    "efésios": "Efésios",
    "filipenses": "Filipenses",
    "colossenses": "Colossenses",
    "hebreus": "Hebreus",
    "apocalipse": "Apocalipse",
}


def normalize(s: str) -> str:
    """Normaliza para comparações simples (minúsculas)."""
    return s.strip().lower()


def verse_key(meta: Dict[str, Any]) -> str:
    """Chave estável para deduplicar um versículo."""
    return f"{meta.get('livro')}|{meta.get('capitulo')}|{meta.get('versiculo')}"


def format_ref(meta: Dict[str, Any]) -> str:
    return f"{meta.get('livro')} {meta.get('capitulo')}:{meta.get('versiculo')}"


def detect_book_filter(question: str) -> Optional[str]:
    """
    Se a pergunta menciona explicitamente um livro (ex.: 'Juízes 16'),
    retornamos o nome do livro para filtrar via metadata.
    """
    q = normalize(question)
    # tenta encontrar match por substring
    for k, proper in BOOK_NAMES.items():
        if k in q:
            return proper
    return None


def prefer_testament(question: str) -> Optional[str]:
    """
    Heurística simples:
    - Se mencionar "Jesus", "Cristo", "evangelho", priorize NT
    - Caso contrário, sem preferência
    """
    q = normalize(question)
    if any(x in q for x in ["jesus", "cristo", "evangelho", "apóstolo", "paulo"]):
        return "NT"
    return None


def extract_keywords(question: str) -> List[str]:
    """
    Extrai keywords para busca literal.
    Mudanças importantes:
    - Aceita tokens de tamanho >= 2 (para pegar "fé")
    - Remove stopwords e duplicados
    - Mantém palavras com acento normalmente (Chroma $contains lida bem com isso)
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
# Busca semântica (embeddings)
# ==========================
def semantic_search(
    vs: Chroma,
    question: str,
    top_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    Busca por embeddings.
    Se `where` vier, aplicamos filtro por metadata (ex.: livro="Juízes").
    """
    hits = vs.similarity_search_with_score(question, k=top_k, filter=where or None)
    out: List[Tuple[str, Dict[str, Any], float]] = []
    for doc, score in hits:
        out.append((doc.page_content, doc.metadata, float(score)))
    return out


# ==========================
# Busca literal (keywords) com ranking
# ==========================
def literal_search_ranked(
    vs: Chroma,
    keywords: List[str],
    per_kw_limit: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], int]]:
    """
    Busca literal: pega candidatos por $contains para cada keyword e
    re-ranqueia pelo número de keywords presentes no texto do versículo.

    Retorna (texto, meta, match_count).
    """
    if not keywords:
        return []

    # limita nº de keywords para não explodir custo
    keywords = keywords[:6]

    # Coletar candidatos (pool)
    pool: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # verse_key -> (text, meta)

    for kw in keywords:
        # where_document filtra pelo conteúdo do texto
        # where (metadata) filtra por livro/testamento quando aplicável
        r = vs._collection.get(
            where=where,
            where_document={"$contains": kw},
            limit=per_kw_limit,
        )
        docs = r.get("documents", []) or []
        metas = r.get("metadatas", []) or []

        for text, meta in zip(docs, metas):
            k = verse_key(meta)
            if k not in pool:
                pool[k] = (text, meta)

    # Re-ranqueia: conta quantas keywords aparecem no texto
    ranked: List[Tuple[str, Dict[str, Any], int]] = []
    for _k, (text, meta) in pool.items():
        t = normalize(text)
        match_count = sum(1 for kw in keywords if kw in t)
        if match_count > 0:
            ranked.append((text, meta, match_count))

    # Ordena: mais matches primeiro
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# ==========================
# Mesclar, deduplicar, priorizar
# ==========================
def merge_results(
    lit: List[Tuple[str, Dict[str, Any], int]],
    sem: List[Tuple[str, Dict[str, Any], float]],
    prefer_nt: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Mescla resultados literal + semântico.
    Preferência:
    - literal primeiro (porque tende a ser mais exato quando acerta)
    - depois semântico (para cobrir perguntas "como/o que é")

    Se prefer_nt == "NT", damos um empurrão: colocamos resultados NT antes
    (sem re-ranquear demais, só uma preferência de exibição).
    """
    merged: List[Tuple[str, Dict[str, Any], str]] = []
    seen = set()

    def add(text: str, meta: Dict[str, Any], source: str):
        k = verse_key(meta)
        if k in seen:
            return
        seen.add(k)
        merged.append((text, meta, source))

    # 1) literal primeiro
    for text, meta, _mc in lit:
        add(text, meta, "literal")

    # 2) semântico depois
    for text, meta, _score in sem:
        add(text, meta, "semantic")

    # 3) se quiser preferir NT (quando pergunta é sobre Jesus), reordena leve
    if prefer_nt == "NT":
        merged.sort(key=lambda x: 0 if x[1].get("testamento") == "NT" else 1)

    return merged


def main() -> int:
    if len(sys.argv) < 2:
        print('Uso: python retrieval/query.py "sua pergunta"')
        return 1

    question = sys.argv[1]

    # Inicializa embeddings + vectorstore
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Detecta filtros úteis
    book = detect_book_filter(question)           # ex.: "Juízes"
    prefer_nt = prefer_testament(question)        # ex.: "NT" se falar de Jesus

    where: Dict[str, Any] = {}
    if book:
        where["livro"] = book
    # Obs.: preferência de NT aqui é só na ordenação final, mas você pode filtrar também.
    # Se quiser filtrar de verdade quando falar de Jesus:
    # if prefer_nt == "NT": where["testamento"] = "NT"

    # Keywords para literal (melhoradas)
    keywords = extract_keywords(question)

    # 1) Literal (ranked)
    lit = literal_search_ranked(vs, keywords, LITERAL_PER_KEYWORD_LIMIT, where=where or None)

    # 2) Semântico (com filtro se houver)
    sem = semantic_search(vs, question, SEMANTIC_TOP_K, where=where or None)

    # 3) Mescla
    merged = merge_results(lit, sem, prefer_nt=prefer_nt)

    if not merged:
        print("Nenhum trecho encontrado.")
        return 0

    # Debug útil (para você entender o comportamento)
    print("\n=== RETRIEVAL HÍBRIDO (GERAL) ===")
    print(f"Pergunta: {question}")
    print(f"Filtro livro: {book or '-'} | Preferência: {prefer_nt or '-'}")
    print(f"Keywords: {keywords}")
    print(f"Literal candidates (dedup/ranked): {len(lit)} | Semântico: {len(sem)} | Final: {len(merged)}")
    print("\n=== TRECHOS RECUPERADOS ===\n")

    for i, (text, meta, source) in enumerate(merged[:FINAL_TOP_N], start=1):
        print(f"[{i}] {format_ref(meta)} | {meta.get('testamento')} | fonte={source}")
        print(text)
        print("-" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
