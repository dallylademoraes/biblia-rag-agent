"""
Ingestion RAG: TXT -> Chroma DB
"""
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

RAW_TXT = Path("data/raw/biblia.txt")
CHROMA_DIR = Path("data/processed/chroma_db_txt")
COLLECTION_NAME = "biblia_almeida_rc_txt"

EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

BOOK_LINE_RE = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+$")          # GÊNESIS
CHAPTER_LINE_RE = re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+)\s+(\d+)$")  # GÊNESIS 1
VERSE_LINE_RE = re.compile(r"^(\d{1,3})\s+(.+)$")                               # 1 No princípio...

def parse_to_documents(path: Path) -> list[Document]:
    testamento = None  # "AT" ou "NT"
    livro = None
    capitulo = None

    docs: list[Document] = [] 

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith("BÍBLIA SAGRADA") or upper.startswith("TRADUÇÃO:") or upper.startswith("EDIÇÃO"):
                continue

            if upper == "ANTIGO TESTAMENTO":
                testamento = "AT"
                continue
            if upper == "NOVO TESTAMENTO":
                testamento = "NT"
                continue

            m = CHAPTER_LINE_RE.match(upper)
            if m:
                livro = m.group(1).title()
                capitulo = int(m.group(2))
                continue

            if BOOK_LINE_RE.match(upper) and not any(ch.isdigit() for ch in upper):
                livro = upper.title()
                capitulo = None
                continue

            vm = VERSE_LINE_RE.match(line)
            if vm and testamento and livro and capitulo:
                versiculo = int(vm.group(1))
                texto = vm.group(2).strip()

                docs.append(
                    Document(
                        page_content=texto,
                        metadata={
                            "testamento": testamento,
                            "livro": livro,
                            "capitulo": capitulo,
                            "versiculo": versiculo,
                            "fonte": path.name,
                            "traducao": "Almeida Revista e Corrigida",
                        },
                    )
                )

    return docs

def main():
    if not RAW_TXT.exists():
        raise FileNotFoundError(f"Não achei: {RAW_TXT.resolve()}")

    docs = parse_to_documents(RAW_TXT)
    print(f"Versículos parseados: {len(docs)}")

    # Embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings.embed_query("ping")
    
     # Vector store
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    
    # Ingestão em lote simples (para não estourar memória)
    batch_size = 128
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vs.add_documents(batch)
        if i == 0 or (i // batch_size) % 50 == 0:
            print(f"Inseridos: {min(i + batch_size, len(docs))}/{len(docs)}")

    print("Ingestão concluída.")
    print(f"Chroma em: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
