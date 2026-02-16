from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Service URLs (inside Minikube namespace)
OLLAMA_URL = "http://ollama:11434"
VECTOR_STORE_URL = "http://vector-store:5002"
EMBEDDING_URL = "http://embedding-service:5001/embed"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a question."})

    # 1️⃣ Get embedding from embedding service
    embed_resp = requests.post(EMBEDDING_URL, json={"text": question})
    embedding = embed_resp.json().get("embedding", [])
    if not embedding:
        return jsonify({"answer": "Failed to get embedding."})

    # 2️⃣ Query vector store for top-k similar documents
    vector_resp = requests.post(
        f"{VECTOR_STORE_URL}/search",
        json={"embedding": embedding, "k": 5}
    )
    top_docs = vector_resp.json().get("results", [])

    # 3️⃣ Build RAG prompt
    context_text = "\n".join(top_docs) if top_docs else ""
    prompt = f"Context: {context_text}\n\nQuestion: {question}"

    # 4️⃣ Call Ollama LLM
    llm_resp = requests.post(
        f"{OLLAMA_URL}/v1/generate",
        json={"model": "qwen-0.5b", "prompt": prompt}
    )
    answer = llm_resp.json().get("text", "Sorry, no answer returned.")

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
