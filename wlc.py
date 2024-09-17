import json
import os
import re
# import time
import numpy as np
import ollama

norm = np.linalg.norm

def parse_file(filename):
    with open(filename, encoding="utf-8") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip().replace("_","")
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings


def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist bs
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings json
    with open(f"embeddings/{filename}.json", 'w') as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", 'r') as f:
        return json.load(f)


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def main():
    # filename = "city-of-god.txt"
    filename = "./wlc.md"
    paragraphs = parse_file(filename)

    # start = time.perf_counter()

    # for p in paragraphs[:10]:
    #     print(p + '\n')
    # print(len(paragraphs))

    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
    # print(len(embeddings))

    # prompt = "What is the cheif end of man?"
    prompt = input("\nAsk the WLC --> ")
    prompt_embedding = ollama.embeddings(model='nomic-embed-text', prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:10]
    # print(most_similar_chunks)
    print("\n".join(paragraphs[item[1]] for item in most_similar_chunks))
    # for item in most_similar_chunks:
    #     print(item[0], paragraphs[item[1]])

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
                "content": SYSTEM_PROMPT + "\nQuestion ".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    print("\n" + re.sub(r'^ ',"",response["message"]["content"]) + "\n")
    # print(response)

    # print(time.perf_counter() - start)



if __name__ == "__main__":
    main()

