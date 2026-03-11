from langchain_community.document_loaders import TextLoader

def load_text():
    loader = TextLoader("sample.txt", encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        print(doc.page_content)

if __name__ == "__main__":
    load_text()