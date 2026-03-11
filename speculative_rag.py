from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -----------------------------
# Step 0: Load LLM
# -----------------------------

print("Step 0: Loading LLM...")

llm = Ollama(model="llama3")

print("Step 0 Completed ✅")

# -----------------------------
# Step 1: Load Embedding Model
# -----------------------------

print("Step 1: Loading Embedding Model...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Step 1 Completed ✅")

# -----------------------------
# Step 2: Load Vector DB
# -----------------------------

print("\nStep 2: Loading Vector Database...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever()

print("Step 2 Completed ✅")

# -----------------------------
# Step 3: Ask Question
# -----------------------------

query = input("\nAsk question: ")

# -----------------------------
# Step 4: Speculative Answer
# -----------------------------

print("\nStep 3: Generating Speculative Answer...")

prompt = f"""
Give a possible speculative answer for the following question.

Question:
{query}

Speculative Answer:
"""

speculative_answer = llm.invoke(prompt)

print("\nSpeculative Answer:")
print(speculative_answer)

print("\nStep 3 Completed ✅")

# -----------------------------
# Step 5: Retrieve Documents
# -----------------------------

print("\nStep 4: Retrieving Documents...")

docs = retriever.invoke(query)

print("Step 4 Completed ✅")

# -----------------------------
# Step 6: Verification
# -----------------------------

print("\nStep 5: Verifying with documents...")

context = ""

for i, doc in enumerate(docs[:3]):

    print(f"\nChunk {i+1}\n")
    print(doc.page_content)

    context += doc.page_content + "\n"

print("\nStep 5 Completed ✅")

# -----------------------------
# Step 7: Final Answer
# -----------------------------

print("\nStep 6: Generating Final Answer with Documents...")

prompt = f"""
Use the following documents to verify and answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

final_answer = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(final_answer)

print("\nSpeculative RAG Pipeline Finished 🚀")

#python speculative_rag.py