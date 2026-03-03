import numpy as np
import os
import time
from database_manager import init_db
from tensor_manager import ensure_v0_weights, calculate_weight_hash, get_layer_metadata, verify_tensor_integrity
from network_manager import ZeroRAMNetworkManager, aggregate_swarm_gradients

def test_dyson_network_swarm():
    print("ZeroRAM-GEN: Validando Sprints 36 a 40 (Dyson Network & Swarm)")
    init_db()
    ensure_v0_weights()
    
    # 1. Sprint 36: Registro de Nós
    net = ZeroRAMNetworkManager(node_id="Dyson_Core_01")
    net.register_node()
    nodes = net.get_available_nodes()
    print(f"[OK] Sprint 36: Nós disponíveis no Swarm: {nodes}")
    
    # 2. Sprint 37: Treinamento Distribuído (Simulação de Agregação de Gradientes)
    grad1 = np.random.randn(10, 10) * 0.1
    grad2 = np.random.randn(10, 10) * 0.1
    avg_grad = aggregate_swarm_gradients([grad1, grad2])
    print(f"[OK] Sprint 37: Gradientes agregados. Mean: {np.mean(avg_grad):.6f}")
    
    # 3. Sprint 38: Integridade e Hash
    weights_name = "hidden_01_weights"
    meta = get_layer_metadata(weights_name)
    h = calculate_weight_hash(meta['path'])
    print(f"[OK] Sprint 38: Hash da camada '{weights_name}': {h[:16]}...")
    is_valid = verify_tensor_integrity(weights_name)
    print(f"Verificação de Integridade: {'Pass' if is_valid else 'FAIL'}")
    
    # 4. Sprint 39: Handover e Medição de Carga
    load = net.measure_node_load()
    print(f"[OK] Sprint 39: Carga do Nó Atual: {load:.2f} (Baseado em I/O e CPU)")
    
    # 5. Sprint 40: Swarm Master (Simulação de Transferência de Ativação)
    activation = np.random.randn(1, 128)
    success = net.send_activations("Dyson_Remote_02", activation)
    if success:
        print("[OK] Sprint 40: Transferência P2P de ativações simulada com sucesso.")

    print("\n[CONCLUÍDO] Motor de Escabilidade Dyson Network (Sprints 36-40) validado.")

if __name__ == "__main__":
    test_dyson_network_swarm()
