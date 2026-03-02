import asyncio
import aiohttp
import random
from typing import List, Set

class GossipService:
    def __init__(self, my_address: str):
        self.my_address = my_address
        self.known_peers: Set[str] = set()
        self.is_running = False

    async def start(self):
        self.is_running = True
        print(f"[Gossip] Iniciando serviço P2P em {self.my_address}")
        asyncio.create_task(self.gossip_loop())

    async def stop(self):
        self.is_running = False

    def add_peer(self, peer_address: str):
        if peer_address != self.my_address:
            self.known_peers.add(peer_address)

    async def gossip_loop(self):
        while self.is_running:
            await asyncio.sleep(10) # Intervalo de Gossip
            if not self.known_peers:
                continue

            # Seleciona um vizinho aleatório para trocar informações
            target = random.choice(list(self.known_peers))
            try:
                async with aiohttp.ClientSession() as session:
                    # Envia lista de vizinhos conhecidos (Sync)
                    payload = {"known_peers": list(self.known_peers)}
                    async with session.post(f"{target}/api/node/sync", json=payload, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            new_peers = data.get("known_peers", [])
                            for p in new_peers:
                                self.add_peer(p)
            except Exception as e:
                print(f"[Gossip] Erro ao sincronizar com {target}: {str(e)}")
