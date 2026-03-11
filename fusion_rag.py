from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

print("Step 0: Loading LLM (Ollama)...")

llm = Ollama(model="llama3")

print("Step 0 Completed ✅")

print("Step 1: Loading Embedding Model...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Step 1 Completed ✅")

print("\nStep 2: Loading Vector Database...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever()

print("Step 2 Completed ✅")

# User query
query = input("\nAsk question: ")

print("\nStep 3: Generating multiple queries...")

queries = [
    query,
    query + " explanation",
    "explain " + query,
    "concept of " + query
]

print("Generated Queries:")
for q in queries:
    print("-", q)

print("\nStep 3 Completed ✅")

print("\nStep 4: Retrieving chunks for each query...")

all_docs = []

for q in queries:
    docs = retriever.invoke(q)
    all_docs.extend(docs)

print("Step 4 Completed ✅")

print("\nStep 5: Removing duplicate chunks...")

unique_text = set()
unique_docs = []

for doc in all_docs:
    if doc.page_content not in unique_text:
        unique_text.add(doc.page_content)
        unique_docs.append(doc)

print("Step 5 Completed ✅")

print("\nStep 6: Final fused results\n")

context = ""

for i, doc in enumerate(unique_docs[:5]):
    print(f"\nChunk {i+1}\n")
    print(doc.page_content)
    context += doc.page_content + "\n"

print("\nStep 6 Completed ✅")

# -------------------------
# Step 7: LLM Answer
# -------------------------

print("\nStep 7: Generating final answer using Ollama...\n")

prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("Final AI Answer:\n")
print(response)

print("\nFusion RAG Pipeline Completed 🚀")

#python fusion_rag.py