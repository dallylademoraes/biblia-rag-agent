"""
retrieval/answer.py - Cohere Embeddings + Groq Chat
"""
import os
import sys
from functools import lru_cache

# 1. Carregar a chave 
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Configurações
CHROMA_DIR = "data/processed/chroma_db_txt"
COLLECTION_NAME = "biblia_almeida_rc_txt"
EMBED_MODEL = "embed-multilingual-v3.0"  # Cohere multilingual
CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # Groq
FALLBACK_CHAT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

def _candidate_chat_models():
    models = []
    env_model = os.getenv("GROQ_MODEL")
    if env_model and env_model.strip():
        models.append(env_model.strip())
    models.append(CHAT_MODEL)
    models.extend(FALLBACK_CHAT_MODELS)

    # de-dup preservando a ordem
    seen = set()
    deduped = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped

def _is_decommissioned_model_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("model_decommissioned" in msg) or ("has been decommissioned" in msg)

@lru_cache(maxsize=1)
def _get_vectorstore():
    """Carrega o vectorstore do Chroma com Cohere Embeddings"""
    embeddings = CohereEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

@lru_cache(maxsize=8)
def _get_llm(model: str):
    """Carrega o modelo Groq"""
    return ChatGroq(
        model=model,
        temperature=0,
        max_tokens=1024
    )

def retrieve_verses(question: str, k=10):
    """Busca versículos relevantes usando Cohere Embeddings"""
    vs = _get_vectorstore()
    docs = vs.similarity_search(question, k=k)
    return docs

def answer(question: str):
    """Responde pergunta usando RAG (Cohere busca + Groq responde)"""
    
    # 1. Busca (Retrieval com Cohere)
    docs = retrieve_verses(question)
    if not docs:
        return "Não encontrei versículos sobre isso."
    
    # 2. Monta o contexto
    context_text = "\n\n".join([
        f"[{d.metadata.get('livro')} {d.metadata.get('capitulo')}:{d.metadata.get('versiculo')}] {d.page_content}" 
        for d in docs
    ])
    
    # 3. Gera a resposta (Generation com Groq)
    msg = HumanMessage(content=f"""
    Use APENAS o contexto abaixo para responder à pergunta.
    Se a resposta não estiver no contexto, diga que não sabe.
    Seja objetivo e cite os versículos relevantes.
    
    CONTEXTO:
    {context_text}
    
    PERGUNTA: {question}
    """)
    
    last_err = None
    for model in _candidate_chat_models():
        try:
            llm = _get_llm(model)
            response = llm.invoke([msg])
            return response.content
        except Exception as e:
            last_err = e
            if _is_decommissioned_model_error(e):
                continue
            raise

    raise last_err

def main():
    if len(sys.argv) < 2:
        print('❌ Uso: python retrieval/answer.py "Sua pergunta"')
        print()
        print('Exemplos:')
        print('  python retrieval/answer.py "Quem foi Moisés?"')
        print('  python retrieval/answer.py "O que a Bíblia diz sobre amor?"')
        print('  python retrieval/answer.py "Onde Jesus nasceu?"')
        return
    
    question = sys.argv[1]
    print(f"🔍 Buscando resposta para: '{question}'...")
    print()
    print("=" * 60)
    print()
    
    try:
        result = answer(question)
        print(result)
    except Exception as e:
        print(f"❌ Erro: {e}")
        print()
        print("💡 Verifique:")
        print("   - Banco Chroma foi criado (rode ingest.py primeiro)")
        print("   - COHERE_API_KEY está no .env")
        print("   - GROQ_API_KEY está no .env")
        print("   - (Opcional) GROQ_MODEL no .env (ex: llama-3.3-70b-versatile)")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
