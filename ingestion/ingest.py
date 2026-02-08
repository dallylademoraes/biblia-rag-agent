"""
Ingestion RAG: TXT -> Chroma DB (com Cohere Embeddings)
"""
import os
from pathlib import Path
import re
import shutil
import time
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

# Configurações
RAW_TXT = Path("data/raw/biblia.txt")
CHROMA_DIR = Path("data/processed/chroma_db_txt")
COLLECTION_NAME = "biblia_almeida_rc_txt"

# Cohere Embeddings - Multilingual (ótimo para português!)
EMBED_MODEL = "embed-multilingual-v3.0"

# Configurações de Rate Limit (Cohere permite 1000/min)
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 5  # 5 segundos (mais rápido que Google!)
MAX_RETRIES = 3

BOOK_LINE_RE = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+$")
CHAPTER_LINE_RE = re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇÜ][A-ZÁÉÍÓÚÂÊÔÃÕÇÜ\s\-]+)\s+(\d+)$")
VERSE_LINE_RE = re.compile(r"^(\d{1,3})\s+(.+)$")

def parse_to_documents(path: Path) -> List[Document]:
    testamento = None
    livro = None
    capitulo = None
    docs: List[Document] = [] 

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

def add_documents_with_retry(vs, batch, batch_num, total_batches):
    """Adiciona documentos com retry em caso de rate limit"""
    for attempt in range(MAX_RETRIES):
        try:
            vs.add_documents(batch)
            print(f"✅ Batch {batch_num}/{total_batches} inserido ({len(batch)} docs)")
            return True
        except Exception as e:
            error_msg = str(e)
            
            # Detectar rate limit
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = DELAY_BETWEEN_BATCHES * (attempt + 2)
                print(f"⏳ Rate limit! Aguardando {wait_time}s... (tentativa {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"❌ Erro: {error_msg}")
                if attempt < MAX_RETRIES - 1:
                    print(f"   Tentando novamente em 5s...")
                    time.sleep(5)
                else:
                    return False
    
    print(f"❌ Falha após {MAX_RETRIES} tentativas")
    return False

def main():
    print("=" * 60)
    print("🚀 INGESTÃO RAG - BÍBLIA SAGRADA")
    print("   Cohere Embeddings (Multilingual) + Chroma DB")
    print("=" * 60)
    print()
    
    if not RAW_TXT.exists():
        raise FileNotFoundError(f"❌ Não achei: {RAW_TXT.resolve()}")

    # Verificar API Key
    if not os.getenv("COHERE_API_KEY"):
        print("❌ ERRO: COHERE_API_KEY não encontrada no .env")
        print("   1. Crie conta em: https://dashboard.cohere.com/")
        print("   2. Copie sua API key")
        print("   3. Adicione no .env: COHERE_API_KEY=sua_chave")
        return

    if CHROMA_DIR.exists():
        print(f"🗑️  Removendo banco antigo em {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    print("📖 Parseando arquivo da Bíblia...")
    docs = parse_to_documents(RAW_TXT)
    print(f"✅ Versículos parseados: {len(docs)}\n")
    
    print(f"🤖 Configurando Cohere Embeddings...")
    print(f"   Modelo: {EMBED_MODEL}")
    
    try:
        embeddings = CohereEmbeddings(
            model=EMBED_MODEL,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        # Teste de conexão
        print("🔌 Testando conexão com Cohere...")
        embeddings.embed_query("teste")
        print("✅ Conexão OK!\n")
        
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        print("\n💡 Verifique:")
        print("   - API key está correta no .env")
        print("   - Você tem conta ativa no Cohere")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    
    # Calcular estimativas
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_time = (total_batches * DELAY_BETWEEN_BATCHES) / 60
    
    print("⚙️  CONFIGURAÇÃO DA INGESTÃO")
    print("-" * 60)
    print(f"   📊 Total de documentos: {len(docs)}")
    print(f"   📦 Tamanho do batch: {BATCH_SIZE}")
    print(f"   🔢 Total de batches: {total_batches}")
    print(f"   ⏱️  Delay entre batches: {DELAY_BETWEEN_BATCHES}s")
    print(f"   🕐 Tempo estimado: ~{estimated_time:.1f} minutos")
    print("-" * 60)
    print()
    
    input("⏸️  Pressione ENTER para iniciar a ingestão...")
    print()
    
    print("🚀 INICIANDO INGESTÃO...")
    print()
    
    start_time = time.time()
    successful = 0
    
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        if add_documents_with_retry(vs, batch, batch_num, total_batches):
            successful += len(batch)
        
        # Delay entre batches (exceto no último)
        if i + BATCH_SIZE < len(docs):
            time.sleep(DELAY_BETWEEN_BATCHES)

    elapsed_time = (time.time() - start_time) / 60
    
    print()
    print("=" * 60)
    print("🎉 INGESTÃO CONCLUÍDA!")
    print("-" * 60)
    print(f"   ✅ Documentos inseridos: {successful}/{len(docs)}")
    print(f"   ⏱️  Tempo total: {elapsed_time:.1f} minutos")
    print(f"   💾 Banco salvo em: {CHROMA_DIR.resolve()}")
    print("=" * 60)
    print()
    print("💡 Próximo passo: python retrieval/answer.py \"Sua pergunta\"")

if __name__ == "__main__":
    main()