import json
from rag_utils import get_text, get_paragraphs_array, chunk_text_by_sentences, chunker
import os
import re
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def embed_texts_with_chunking(collection, texts_dir, title, code):
    files = []
    for file in os.listdir(texts_dir):
        files.append(file)

    j = 1
    for file in sorted(files): 
        paragraphsArr = get_paragraphs_array(texts_dir + "/" + file)
        chunks = []
        metas = []
        ids = []

        i = 1
        # for p in paragraphsArr[3:20]:
        for p in paragraphsArr[5:]:
            paragraph_chunks = chunk_text_by_sentences(source_text=p, sentences_per_chunk=4, overlap=0 )
            chunks.extend(paragraph_chunks)

            k = 1
            for pc in paragraph_chunks:
                metas.append({"title": title, "book": str(j), "paragraph": str(i)})
                # ids.append(code + "_"+ f"{j:02d}" + "." + f"{i:02d}" + "." + f"{k:02d}")
                ids.append(code + f"{j:02d}" + f"{i:03d}" + f"{k:02d}")
                # print(pc + "\n" + f"[{len(paragraph_chunks)} chunks" + ']\n')
                k += 1
            i += 1
        j += 1

        # for c in chunks:
        # print(c + "\n")
        # print(metas)
        # print(ids)
        # print(str(len(chunks)) + " " + str(len(metas)) + " " + str(len(ids)))
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        # print(collection.peek())

        #
        # for index, chunk in enumerate(chunks):
        #     embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
        #     print(".", end="", flush=True)
        #     collection.add([filename+str(index)], [embed], documents=[chunk], metadatas={"source": filename})

    return collection

def main():
    collection_name = "city-of-god"
    texts_dir = "./city_of_god"
    title = "City of God"
    code = "cog"
    # filename = "./texts/city-of-god.txt"

    cc = chromadb.HttpClient(host='localhost', port=8000)
    # cc.delete_collection(name=collection_name)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )
    collection = cc.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)
    if len(collection.peek()['ids']) == 0:
        embed_texts_with_chunking(collection, texts_dir, title, code)
    # print(collection.peek())
    # print(collection.get(
    #     include=["documents"],
    #     where={"paragraph": "1"}
    # ))


    # prompt = input("\nAsk City of God --> ")
    prompt = "What is wrong with Roman religion?"

    results = collection.query(
        query_texts=[prompt], # Chroma will embed this for you
        n_results=10 # how many results to return
    )
    # print(json.dumps(results, indent=2))

    m = 0
    for r in results["documents"][0]:


        print(results["metadatas"][0][m]["book"] + "   " + r)
        m += 1

    # results = [list(item) for item in zip(results["metadatas"][0],results["documents"][0])]
    # for r in results:
    #     # print(r[0] + "\n" + r[1] + "\n")
    #     print(r)

    # print([list(item) for item in zip(results["metadatas"][0],results["documents"][0])])

    SYSTEM_PROMPT = """You are an expert on early Christianity and Early Church History who answers questions on snippets of text provided in context.
Answer only using the context provided.

Context: """
    # print(SYSTEM_PROMPT + "\n" + "\n\n".join(results["documents"][0]))
    # response = ollama.chat(
    #     model = "llama3",
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": SYSTEM_PROMPT + "\n\n".join(results["documents"][0])
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         },
    #     ],
    # )
    # print("\n" + re.sub(r'^ ',"",response["message"]["content"]) + "\n")


if __name__ == "__main__":
    main()

