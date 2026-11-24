
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# STEP 1: Load and split documents

loader = TextLoader(
    "data/raw/solar_energy.txt",
    encoding="utf-8"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# STEP 2: Initialize embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
)


# STEP 3: Initialize vector store

vector_store = Chroma(
    collection_name="renewable_energy",
    embedding_function=embeddings,
    persist_directory="data/chroma_lesson7"
)

ids = [
    f"solar_chunk_{i}"
    for i in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)


# STEP 4: Create a retriever

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# STEP 5: Create the prompt

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sustainability assistant.

        Answer using only the supplied context.
        Do not invent information.
        If the context does not contain the answer,
        say you do not have enough information.

        Context:
        {context}
        """
    ),
    (
        "human",
        "{question}"
    )
])


# STEP 6: Initialize the LLM

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ),
    temperature=0.2
)


# STEP 7: Format retrieved documents

def format_docs(documents):

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        formatted_documents.append(
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# STEP 8: Build the retrieval chain

retrieval_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


# STEP 9: Ask a question

question = "How can solar energy reduce pollution?"

answer = retrieval_chain.invoke(question)

print("\nQuestion:", question)
print("\nAnswer:", answer)
