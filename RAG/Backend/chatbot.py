
# This file loads the FAISS vector database,
# retrieves relevant healthcare information
# and sends it to the LLM to generate responses.

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS


# Step 1: Load Environment Variables

load_dotenv()


# Step 2: Load the Embedding Model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 3: Load the Saved FAISS Database

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# Step 4: Create the Retriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Step 5: Initialize the LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Step 6: Generate Response Function

def generate_response(question, chat_history=None):

    # Retrieve relevant healthcare documents
    documents = retriever.invoke(question)

    # Convert retrieved documents into context
    context = "\n\n".join(
        [document.page_content for document in documents]
    )

    # Final Prompt
    final_prompt = f"""
    You are MediKnowledge AI, a healthcare and medicine
    information assistant.

    Answer the user's question using only the information
    provided in the retrieved context.

    Instructions:

    1. Provide clear, accurate and easy-to-understand answers.
    2. Use simple language and organize answers properly.
    3. If the question is related to medicines, explain only
       the information available in the documents.
    4. Never independently recommend medicines or dosages.
    5. Do not provide medical diagnosis or prescribe treatment.
    6. If the answer is not available in the context, say:
       "I don't know based on the provided documents."
    7. Do not make up medical information.
    8. If relevant, mention the source document and page number.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate the answer
    response = llm.invoke(final_prompt)

    return response.content

