from dotenv import load_dotenv
import os
load_dotenv()
from consts import INDEX_NAME
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import ReadTheDocsLoader

#DONE

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
	docs_path = r"C:\Users\USUARIO\PycharmProjects\doc-assistant\langchain-docs\api.python.langchain.com\en\latest"
	loader = ReadTheDocsLoader(docs_path, encoding='utf-8', errors='ignore')
	raw_documents = loader.load()
	print(raw_documents)
	print(f"loaded {len(raw_documents)} raw documents")
	text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
	documents = text_splitter.split_documents(raw_documents)
	print(f"loaded {len(documents)} documents")
	for doc in documents :
		new_url = doc.metadata["source"]
		new_url = new_url.replace("langchain-docs", "https:/")
		doc.metadata.update({"source": new_url})

	print(f"Going to add {len(documents)} to Pincecone")
	PineconeVectorStore.from_documents(
		documents, embeddings, index_name= INDEX_NAME)

if __name__ == "__main__":
	ingest_docs()