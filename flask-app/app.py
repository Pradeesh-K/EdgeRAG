from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)
OLLAMA_URL = "http://ollama:11434"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    response = requests.post(
        f"{OLLAMA_URL}/v1/generate",
        json={"model": "qwen-0.5b", "prompt": question}
    )
    text = response.json().get("text", "")
    return jsonify({"answer": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
