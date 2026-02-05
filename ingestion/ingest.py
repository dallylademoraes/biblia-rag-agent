"""
Ingestion RAG: TXT -> Chroma DB 
"""
import os
from pathlib import Path
import re
import shutil # Para apagar a pasta antiga

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings # MUDANÇA AQUI
from langchain_chroma import Chroma

# Configurações
RAW_TXT = Path("data/raw/biblia.txt")
CHROMA_DIR = Path("data/processed/chroma_db_txt")
COLLECTION_NAME = "biblia_almeida_rc_txt"

# Modelo de Embedding do Google
EMBED_MODEL = "models/text-embedding-004" 

BOOK_LINE_RE = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+$")
CHAPTER_LINE_RE = re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+)\s+(\d+)$")
VERSE_LINE_RE = re.compile(r"^(\d{1,3})\s+(.+)$")

def parse_to_documents(path: Path) -> list[Document]:
    testamento = None
    livro = None
    capitulo = None

    docs: list[Document] = [] 

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue

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

    # 1. Limpeza: Apagar o banco antigo para não misturar embeddings
    if CHROMA_DIR.exists():
        print(f"Removendo banco antigo em {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    docs = parse_to_documents(RAW_TXT)
    print(f"Versículos parseados: {len(docs)}")

    # 2. Configurar Embeddings do Google (Pega a chave do ambiente)
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("A variável GOOGLE_API_KEY não está definida!")
        
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    
    # Teste rápido de conexão
    try:
        embeddings.embed_query("teste")
    except Exception as e:
        print(f"Erro ao conectar na API do Google: {e}")
        return

    # 3. Criar Vector Store
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    
    # Ingestão em lote
    batch_size = 100 # Reduzi um pouco por segurança da API
    print("Iniciando ingestão no Google...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vs.add_documents(batch)
        if i == 0 or (i // batch_size) % 10 == 0:
            print(f"Inseridos: {min(i + batch_size, len(docs))}/{len(docs)}")

    print("Ingestão concluída.")

if __name__ == "__main__":
    main()