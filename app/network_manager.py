import os
import json
import time
import requests
from database_manager import get_db_connection, log_telemetry

class ZeroRAMNetworkManager:
    """Gerencia a comunicação entre nós do Swarm (Dyson Network)."""
    
    def __init__(self, node_id="local_node_01"):
        self.node_id = node_id
        self.base_url = "http://localhost:8000" # Simulação
        
    def register_node(self):
        """Registra o nó local no banco de dados de rede."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO network_nodes (node_id, base_url, status) VALUES (?, ?, ?)",
                (self.node_id, self.base_url, "online")
            )
            conn.commit()
        print(f"[NET] Nó '{self.node_id}' registrado no Swarm.")

    def get_available_nodes(self):
        """Lista todos os nós online no Swarm."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT node_id, base_url FROM network_nodes WHERE status = 'online'")
            return cursor.fetchall()

    def send_activations(self, target_node_id, activations):
        """Simula o envio de ativações para outro nó."""
        print(f"[NET] Enviando ativações ({activations.shape}) para {target_node_id}...")
        # Em uma implementação real, usaríamos requests.post ou WebSockets
        log_telemetry('network_transfer_size', activations.nbytes, f"to:{target_node_id}")
        return True

    def measure_node_load(self):
        """Mede a carga do nó (CPU, RAM e Latência de I/O de disco)."""
        import psutil
        cpu_usage = psutil.cpu_percent()
        # Simula medição de latência de leitura recente da telemetria
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(value) FROM telemetry WHERE metric_name = 'io_read_latency' ORDER BY timestamp DESC LIMIT 10")
            avg_io_latency = cursor.fetchone()[0] or 0.0
            
        load_score = (cpu_usage * 0.3) + (avg_io_latency * 1000 * 0.7)
        return load_score

def aggregate_swarm_gradients(grad_list):
    """Simula a agregação de gradientes de múltiplos nós (All-reduce)."""
    if not grad_list:
        return None
    
    print(f"[SWARM] Agregando gradientes de {len(grad_list)} nós...")
    # Média simples dos gradientes
    avg_grad = sum(grad_list) / len(grad_list)
    return avg_grad
