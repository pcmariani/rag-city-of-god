# import json
import re
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def parse_file(filename):
    with open(filename, encoding="utf-8") as f:
        documents = []
        buffer = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                buffer.append(stripped_line)
            elif buffer:
                documents.append(" ".join(buffer))
                buffer = []
    if buffer:
        documents.append(" ".join(buffer))
    return documents

def parse_raw_documents(raw_documents): 
    documents = []
    ids = []
    metadatas = []
    current_topic = 'FUNDAMENTAL DOCTRINES'
    for doc in raw_documents:
        if re.match('^[^\\s\\d](?!\\.).*', doc):
            current_topic = doc
            continue
        documents.append("Question " + doc.replace("_","").replace("  ", " "))
        metadata = {}
        metadata["topic"] = current_topic
        id = re.search(r'\d+', doc)
        if id: 
            metadata["question_number"] = id.group()
            ids.append("WLC" + id.group())
        metadatas.append(metadata)
    # for d in documents:
    #     print(d)
    # for m in metadatas:
    #     print(m)
    # for id in ids:
    #     print(id)
    # print(len(documents))
    # print(len(ids))
    # print(len(metadatas))
    return [documents, ids, metadatas]

def manage_collection(collection_name, filename):
    cc = chromadb.HttpClient(host='localhost', port=8000)
    # cc.delete_collection(name=collection_name)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )
    collection = cc.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)
    if len(collection.peek()['ids']) == 0:
        raw_documents = parse_file(filename)
        documents, ids, metadatas = parse_raw_documents(raw_documents)
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        # print(collection.peek())
    return collection

def main():
    collection_name = "my_collection"
    filename = "./wlc.md"
    prompt = input("\nAsk the WLC --> ")

    collection = manage_collection(collection_name, filename)
    results = collection.query(
        query_texts=[prompt], # Chroma will embed this for you
        n_results=5 # how many results to return
    )
    # print(json.dumps(results, indent=2))
    # print(type(results))

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as consize as possible. If you're unsure, just say that you don't konw.
        Context:

    """
    response = ollama.chat(
        model = "llama3",
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\n".join(results["documents"][0])
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    print("\n" + re.sub(r'^ ',"",response["message"]["content"]) + "\n")


if __name__ == "__main__":
    main()

