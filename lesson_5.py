
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Load the document

loader = TextLoader(
    "data/raw/practice.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Number of documents:", len(documents))

print("\nOriginal document:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)


# STEP 2: Create a text splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# STEP 3: Split the documents

chunks = splitter.split_documents(documents)

print("\nNumber of chunks:", len(chunks))


# STEP 4: Inspect each chunk

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")

    print("Text:")
    print(chunk.page_content)

    print("Metadata:")
    print(chunk.metadata)

    print("Character count:", len(chunk.page_content))
