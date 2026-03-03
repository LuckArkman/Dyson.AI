from flask import Flask, request, jsonify, Response
import numpy as np
import time
from engine import embedding_lookup, dense_layer_forward, compute_softmax
from database_manager import get_or_create_id, get_text_by_id, init_db
from tensor_manager import load_model_checkpoint

app = Flask(__name__)

# Garantir que o modelo treinado esteja carregado
init_db()
load_model_checkpoint("v1.0_trained")

def generate_text(prompt, max_tokens=10):
    """Lógica simplificada de geração autoregressiva para o servidor."""
    token_ids = [get_or_create_id(w) for w in prompt.split()]
    generated_ids = []
    
    current_ids = token_ids
    for _ in range(max_tokens):
        # Forward pass (Zero RAM)
        emb = embedding_lookup(current_ids)
        x = np.mean(emb, axis=0, keepdims=True)
        h1 = dense_layer_forward(x, "hidden_01_weights", "hidden_01_bias", activation='relu')
        logits = dense_layer_forward(h1, "output_weights", activation='linear')
        probs = compute_softmax(logits)
        
        # Escolher o próximo token (Greedy Search)
        next_id = int(np.argmax(probs[0]))
        generated_ids.append(next_id)
        
        # Parar se for endoftext
        if next_id == 4: # <|endoftext|>
            break
            
        current_ids = current_ids + [next_id]
        if len(current_ids) > 16: # Janela de contexto curta para o demo
            current_ids = current_ids[-16:]
            
    return " ".join([get_text_by_id(tid) for tid in generated_ids])

@app.route("/api/v1/generate", methods=["POST"])
def api_generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 10)
    
    start_time = time.time()
    response_text = generate_text(prompt, max_tokens)
    latency = time.time() - start_time
    
    return jsonify({
        "status": "stable",
        "model": "ZeroRAM-GEN-v1.0",
        "prompt": prompt,
        "response": response_text,
        "latency_sec": latency
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready", "memory": "ZeroRAM Principle Optimized"})

if __name__ == "__main__":
    print("ZeroRAM Inference Server - Iniciando em http://localhost:8000")
    app.run(port=8000, debug=False, threaded=True)
