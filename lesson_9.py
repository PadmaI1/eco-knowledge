from pathlib import Path
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
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

documents = []

for file_path in Path("data/raw").glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents")
print(f"Created {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

prompt = ChatPromptTemplate.from_template("""
You are an environmental knowledge assistant.

Answer the question using ONLY the information in the
provided context.

If the context does not contain enough information,
clearly say that you don't have enough information.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

answer_chain = prompt | model | StrOutputParser()


def ask_question(question):

    # 1. Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # 2. Handle the case where nothing was retrieved
    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant information.",
            "sources": []
        }

    # 3. Combine the retrieved document contents
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # 4. Send context and question through the LLM chain
    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })

    # 5. Extract source filenames
    sources = list(set(
        doc.metadata.get("source", "Unknown")
        for doc in retrieved_docs
    ))

    # 6. Return both the answer and sources
    return {
        "answer": answer,
        "sources": sources
    }

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    result = ask_question(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)