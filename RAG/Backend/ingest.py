
# This file loads healthcare PDFs, splits the documents,
# generates embeddings and stores them in FAISS vector database.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

# Step 1: Define PDF file paths

pdf_files = [
    "data/Health_Hygiene.pdf",
    "data/Classification_of_Drugs.pdf",
    "data/NLEM.pdf"
]

# Step 2: Load all PDF documents

documents = []

for pdf_file in pdf_files:

    loader = PyPDFLoader(pdf_file)

    pdf_documents = loader.load()

    documents.extend(pdf_documents)

    print(f"Loaded: {pdf_file}")

print("Total pages loaded:", len(documents))


# Step 3: Split documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# Step 4: Create the embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 5: Create FAISS vector database

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# Step 6: Save vector database

os.makedirs("vectorstore", exist_ok=True)

vectorstore.save_local("vectorstore")

print("Healthcare Vector Database Created Successfully!")

