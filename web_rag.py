from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from ddgs import DDGS

# -------------------------
# Step 0: Load LLM
# -------------------------
print("Loading LLM...")
llm = OllamaLLM(model="llama3")
print("LLM Loaded ✅")

# -------------------------
# Step 1: Load Embedding
# -------------------------
print("Loading Embedding Model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding Loaded ✅")

# -------------------------
# Step 2: Load Vector Database
# -------------------------
print("Loading Vector Database...")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)
retriever = db.as_retriever(search_kwargs={"k":3})
print("Vector DB Loaded ✅")

# -------------------------
# Step 3: Ask Question
# -------------------------
user_query = input("\nAsk question: ")

# -------------------------
# Step 4: Retrieve from DB
# -------------------------
docs = retriever.invoke(user_query)

# -------------------------
# Step 5: Strict Relevance Check
# -------------------------
stopwords = {"what","is","the","a","an","of","to","for","on","and","current","situation","about","tell"}
keywords = [w.lower() for w in user_query.split() if w.lower() not in stopwords]

relevant_docs = []
for doc in docs:
    content = doc.page_content.lower()
    # Require at least one exact keyword match (spaces prevent substring mismatch)
    if any(f" {k} " in f" {content} " for k in keywords):
        relevant_docs.append(doc)

# -------------------------
# Step 6: Decide DB or Web
# -------------------------
if len(relevant_docs) == 0:
    print("\n❌ DB has no relevant info, searching web...")

    web_content = ""
    with DDGS() as ddgs:
        results = ddgs.text(user_query, max_results=3)
        for r in results:
            print("\n🌐 Web Result")
            print("Title:", r["title"])
            print("Summary:", r["body"])
            web_content += r["body"] + "\n"

    prompt = f"""
Use the following web content to answer the question.

Web Context:
{web_content}

Question:
{user_query}

Answer:
"""
    final_answer = llm.invoke(prompt)
    print("\n🤖 AI Answer (Web Search RAG):\n")
    print(final_answer)

else:
    print("\n📄 DB has relevant info, generating answer...")

    db_context = ""
    for i, doc in enumerate(relevant_docs[:3]):
        print(f"\nChunk {i+1}\n")
        print(doc.page_content)
        db_context += doc.page_content + "\n"

    prompt = f"""
Use the following document context to answer the question.

Context:
{db_context}

Question:
{user_query}

Answer:
"""
    final_answer = llm.invoke(prompt)
    print("\n🤖 AI Answer (DB RAG):\n")
    print(final_answer)

print("\nWeb Search RAG Pipeline Finished 🚀")