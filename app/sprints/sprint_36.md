# Sprint 36: Escabilidade - Etapa 1 (Inferência em Swarm/Rede) [CONCLUÍDA]

## Objetivos
- [x] Implementar a distribuição de camadas em diferentes dispositivos/arquivos (Dyson P2P).
- [x] Desenvolver a rotina de "Transferência de Ativações" simulada.

## Ferramentas & Pacotes
- [x] **network_manager.py**: Gerenciamento de nós e comunicação.
- [x] **SQLite3 (Network)**: Tabela `network_nodes` para descoberta de parceiros.

## Funções e Implementações
- [x] `register_node()`: Entrada do nó local no sistema distribuído.
- [x] `send_activations()`: Simulação de transferência de tensores entre nós.

## Detalhes Técnicos
O ZeroRAM-GEN agora possui os ganchos necessários para operar em rede, permitindo que diferentes discos em máquinas distintas contribuam para uma única inferência.
