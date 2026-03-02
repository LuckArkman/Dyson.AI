from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
import asyncio
import socket

from brain.neural_network_lstm import GenerativeNeuralNetworkLSTM
from gpu.gpu_math_engine import GpuMathEngine
from services.model_trainer import ModelTrainer
from services.gossip_service import GossipService
from core.crypto_utils import CryptoUtils

app = FastAPI(title="Galileu Node API (Python)", version="1.1.0")

# Helpers para detectar endereço IP local
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class NodeState:
    def __init__(self, address: str):
        self.address = address
        self.math_engine = GpuMathEngine()
        self.model: Optional[GenerativeNeuralNetworkLSTM] = None
        self.is_training = False
        self.trainer = ModelTrainer(self.math_engine)
        self.gossip = GossipService(address)
        self.public_key, self.private_key = CryptoUtils.generate_key_pair()

# Configuração de inicialização
PORT = int(os.environ.get("PORT", 8000))
MY_ADDRESS = f"http://{get_local_ip()}:{PORT}"
state = NodeState(MY_ADDRESS)

@app.on_event("startup")
async def startup_event():
    print(f" === Galileu Node (Python) Online em {MY_ADDRESS} ===")
    print(f" === Identidade (Public Key): {CryptoUtils.normalize_public_key(state.public_key)[:20]}... ===")
    await state.gossip.start()

@app.get("/")
def read_root():
    return {
        "status": "Node Online",
        "address": state.address,
        "p2p_peers_count": len(state.gossip.known_peers)
    }

# Rota de Sincronização P2P (Gossip)
@app.post("/api/node/sync")
async def sync_peers(request: Request):
    data = await request.json()
    received_peers = data.get("known_peers", [])
    for p in received_peers:
        state.gossip.add_peer(p)
    return {"known_peers": list(state.gossip.known_peers)}

@app.post("/api/generative/train")
async def train_model():
    if state.is_training: return {"message": "Treinamento em curso."}
    state.is_training = True
    async def task():
        try:
            state.model = GenerativeNeuralNetworkLSTM(1000, 64, 128, "Dayson/pt_0.txt", state.math_engine)
            state.trainer.train_model(state.model, "Dayson/pt_0.txt", "Dayson/Dayson.json", 0.001, 25, 32, 12, 0.1)

        finally: state.is_training = False
    asyncio.create_task(task())
    return {"message": "Treinamento disparado."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
