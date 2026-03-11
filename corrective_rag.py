from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

print("Step 1: Loading LLM...")

# Ollama LLM
generator = Ollama(model="llama3")

print("Step 1 Completed ✅")

print("\nStep 2: Loading Embedding Model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Step 2 Completed ✅")

print("\nStep 3: Loading Vector Database...")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever()
print("Step 3 Completed ✅")

# User question
query = input("\nAsk question: ")

# ----------------------------
# Step 4: LLM Generated Answer
# ----------------------------
print("\nStep 4: LLM generating answer...")

llm_answer = generator.invoke(query)

print("\nLLM Answer:")
print(llm_answer)

print("\nStep 4 Completed ✅")

# ----------------------------
# Step 5: Retrieve from Vector DB
# ----------------------------
print("\nStep 5: Retrieving from Vector DB...")

docs = retriever.invoke(query)

retrieved_text = ""

for doc in docs[:3]:
    retrieved_text += doc.page_content + "\n"

print("\nRetrieved Answer:")
print(retrieved_text)

print("\nStep 5 Completed ✅")

# ----------------------------
# Step 6: Combine Answers
# ----------------------------
print("\nStep 6: Combining LLM + Retrieved Answer...")

final_answer = f"""
Final Answer (Hybrid RAG)

LLM Knowledge:
{llm_answer}

Document Knowledge:
{retrieved_text}
"""

print(final_answer)

print("\nStep 6 Completed ✅")
print("\nPipeline Finished 🚀")

#python corrective_rag.py