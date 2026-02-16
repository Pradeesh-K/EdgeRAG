from flask import Flask, request, jsonify
import faiss
import numpy as np
import os
import pickle
import requests

app = Flask(__name__)

# FAISS setup
INDEX_PATH = "/data/faiss.index"
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    print("Loaded existing FAISS index")
else:
    dim = 384  # MiniLM-L6-v2 embedding dimension
    index = faiss.IndexFlatL2(dim)
    print("Created new FAISS index")

id_to_text = {}

# URL of embedding service in Kubernetes
EMBEDDING_URL = "http://embedding-service:5001/embed"

@app.route("/add", methods=["POST"])
def add_vector():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1️⃣ Get embedding from embedding service
    try:
        embed_resp = requests.post(EMBEDDING_URL, json={"text": text})
        embedding = embed_resp.json().get("embedding", [])
        if not embedding:
            return jsonify({"error": "Failed to get embedding"}), 500
    except Exception as e:
        return jsonify({"error": f"Embedding service error: {str(e)}"}), 500

    # 2️⃣ Add to FAISS
    embedding_np = np.array(embedding, dtype="float32")
    index.add(np.expand_dims(embedding_np, axis=0))
    vector_id = index.ntotal - 1
    id_to_text[vector_id] = text

    # 3️⃣ Persist FAISS index
    os.makedirs("/data", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    return jsonify({"id": vector_id, "text": text})

@app.route("/search", methods=["POST"])
def search_vector():
    data = request.json
    query_text = data.get("text", "").strip()
    k = data.get("k", 5)
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400

    # Get embedding for query
    try:
        embed_resp = requests.post(EMBEDDING_URL, json={"text": query_text})
        query_embedding = embed_resp.json().get("embedding", [])
        if not query_embedding:
            return jsonify({"error": "Failed to get query embedding"}), 500
    except Exception as e:
        return jsonify({"error": f"Embedding service error: {str(e)}"}), 500

    # Search FAISS
    query_np = np.array(query_embedding, dtype="float32")
    D, I = index.search(np.expand_dims(query_np, axis=0), k)
    results = [id_to_text[i] for i in I[0] if i in id_to_text]

    return jsonify({"results": results})

if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    app.run(host="0.0.0.0", port=5002)
