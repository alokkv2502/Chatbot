import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_directory = os.path.join(base_directory, "pdf-documents")

loader = DirectoryLoader(
    pdf_directory,  # Use the calculated path
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
)
docs = loader.load()

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', )

text_splitter = SemanticChunker(
    embeddings=embeddings
)

# flattened_docs = [doc[0] for doc in docs if doc]
chunks = text_splitter.split_documents(docs)

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="collection164",
    connection_string = "postgresql+psycopg2://alok:Alok1234@localhost:5432/database164",

    pre_delete_collection=True,
)