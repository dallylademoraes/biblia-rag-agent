"""
retrieval/answer.py
"""
import sys
from functools import lru_cache

# 1. Carregar a chave 
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Configurações
CHROMA_DIR = "data/processed/chroma_db_txt"
COLLECTION_NAME = "biblia_almeida_rc_txt"
EMBED_MODEL = "models/text-embedding-004" # O mesmo da ingestão
CHAT_MODEL = "gemini-2.5-flash" # Rápido e barato

@lru_cache(maxsize=1)
def _get_vectorstore():
    # Pega a chave do ambiente automaticamente
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

@lru_cache(maxsize=1)
def _get_llm():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

def retrieve_verses(question: str, k=10):
    vs = _get_vectorstore()
    # Busca por similaridade
    docs = vs.similarity_search(question, k=k)
    return docs

def answer(question: str):
    # 1. Busca (Retrieval)
    docs = retrieve_verses(question)
    if not docs:
        return "Não encontrei versículos sobre isso."
    
    # 2. Monta o contexto
    context_text = "\n\n".join([f"[{d.metadata.get('livro')} {d.metadata.get('capitulo')}:{d.metadata.get('versiculo')}] {d.page_content}" for d in docs])
    
    # 3. Gera a resposta (Generation)
    llm = _get_llm()
    msg = HumanMessage(content=f"""
    Use APENAS o contexto abaixo para responder à pergunta.
    Se a resposta não estiver no contexto, diga que não sabe.
    
    CONTEXTO:
    {context_text}
    
    PERGUNTA: {question}
    """)
    
    response = llm.invoke([msg])
    return response.content

def main():
    if len(sys.argv) < 2:
        print('Uso: python retrieval/answer.py "Sua pergunta"')
        return
    
    question = sys.argv[1]
    print(f"Buscando resposta para: '{question}'...\n")
    print(answer(question))

if __name__ == "__main__":
    main()