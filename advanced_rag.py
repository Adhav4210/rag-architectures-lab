from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ddgs import DDGS

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
# Step 2: Load Vector Database
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

print("\nStep 3: Searching Vector DB...")

docs = retriever.invoke(query)

print("Step 3 Completed ✅")

# -----------------------------
# Step 4: Relevance Check
# -----------------------------

print("\nStep 4: Checking relevance...")

stopwords = {
    "what","is","the","in","a","an","of","to","for","on",
    "and","current","situation","about","tell"
}

keywords = [
    word for word in query.lower().split()
    if word not in stopwords
]

relevant_docs = []

for doc in docs:

    content = doc.page_content.lower()

    if any(keyword in content for keyword in keywords):

        relevant_docs.append(doc)

print("Step 4 Completed ✅")

# -----------------------------
# Step 5: Decision Layer
# -----------------------------

if not relevant_docs:

    print("\n❌ No relevant document answer found")

    print("\nStep 5: Searching Web...")

    web_context = ""

    with DDGS() as ddgs:

        results = ddgs.text(query, max_results=3)

        for r in results:

            print("\n🌐 Web Result")
            print("Title:", r["title"])
            print("Summary:", r["body"])

            web_context += r["body"] + "\n"

    # LLM Web Answer
    prompt = f"""
Use the following web information to answer the question.

Web Context:
{web_context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n🤖 AI Answer (Web Based):\n")
    print(response)

    print("\nStep 5 Completed 🌐")

else:

    print("\n📄 Document Answer Found")

    context = ""

    for i, doc in enumerate(relevant_docs[:3]):

        print(f"\nChunk {i+1}\n")
        print(doc.page_content)

        context += doc.page_content + "\n"

    # LLM Document Answer
    prompt = f"""
Answer the question using the provided document context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n🤖 AI Answer (Document Based):\n")
    print(response)

    print("\nStep 5 Completed 📄")

print("\nAdaptive RAG Pipeline Finished 🚀")

#python advanced_rag.py