import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeLangChain


load_dotenv()

# Obtener las variables necesarias
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("INDEX_NAME")


pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='gcp', region=PINECONE_ENV)
    )

def test_pinecone_search(query: str):
    embeddings = OpenAIEmbeddings()
    
    
    vectorstore = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    print(f"Running search for query: '{query}'")
    docs = vectorstore.similarity_search(query, 3) 

    if docs:
        print(f"Found {len(docs)} documents.")
        for doc in docs:
            print("Document metadata:", doc.metadata)
    else:
        print("No documents found.")

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    print(f"Testing document search for query: '{query}'")
    test_docs = docsearch.similarity_search(query, 3)
    print(f"Found {len(test_docs)} documents.")
    if not test_docs:
        print("No relevant documents found.")
        return {"result": "No relevant documents found.", "source_documents": []}

    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_prompt)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    print("Raw LLM Response:", result)

    new_result = {
        "query": result.get("input", ""),
        "result": result.get("answer", "No answer found."),
        "source_documents": result.get("source_documents", []),
    }

    print("Formatted Result:", new_result)
    return new_result

#DONE

# Ejecuta la búsqueda de prueba
if __name__ == "__main__":
    test_pinecone_search("What is LangChain?")
