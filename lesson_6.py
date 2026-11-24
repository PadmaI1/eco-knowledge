
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# STEP 1: Load the document

loader = TextLoader(
    "data/raw/practice.txt",
    encoding="utf-8"
)

documents = loader.load()


# STEP 2: Split into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# STEP 3: Initialize the embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
)


# STEP 4: Initialize Chroma

vector_store = Chroma(
    collection_name="renewable_energy",
    embedding_function=embeddings,
    persist_directory="data/chroma_lesson7"
)


# STEP 5: Add documents to the vector store

ids = [
    f"solar_chunk_{i}"
    for i in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)

print("Documents added to vector store!")


# STEP 6: Search

question = "How can solar energy reduce pollution?"

results = vector_store.similarity_search(
    question,
    k=3
)

print("\nSearch results:")

for i, document in enumerate(results):

    print(f"\nResult {i + 1}")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
