# Sprint 36: Escabilidade - Etapa 1 (Inferência em Swarm/Rede)

## Objetivos
- Implementar a distribuição de camadas em diferentes dispositivos/arquivos (Dyson P2P).
- Desenvolver a rotina de "Transferência de Ativações" via WebSockets.

## Ferramentas & Pacotes
- **Python (WebSockets/HTTP)**: Comunicação entre instâncias do ZeroRAM-GEN.
- **SQLite3 (Network)**: Tabela `NETWORK_NODES`.

## Funções e Implementações
- `send_activations_to_node(node_id, activation_tensor)`: Enviar resultado do Forward para o próximo nó.
- `receive_and_process_activations(data)`: Capturar resultado da camada anterior e rodar a atual localmente.
- `register_local_node()`: Entrar no Swarm do ZeroRAM-GEN.

## Detalhes Técnicos
O ZeroRAM-GEN agora pode ser esticado por vários discos físicos através da rede (Seção 2 do White Paper).
