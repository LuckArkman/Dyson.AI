# Sprint 39: Handover de Inferência (Transferência de Responsabilidade)

## Objetivos
- Implementar o Handover de inferência entre nós do Swarm.
- Passar a responsabilidade do cálculo da frase para o nó que possuir o menor I/O ou maior CPU no momento.

## Ferramentas & Pacotes
- **Python (psutil, time)**: Medir carga de I/O em tempo real.
- **SQLite3 (Network)**: Tabela `NODE_STATUS`.

## Funções e Implementações
- `measure_node_load()`: Calcular métrica de disponibilidade baseada em DISK_READ_LATENCY.
- `elect_best_node_for_layer(layer_index)`: Escolher o nó com o disco mais rápido para a próxima camada.
- `transfer_activation_and_handover(node_id)`: Enviar ativações e ceder a vez de cálculo.

## Detalhes Técnicos
Esta sprint encerra o motor de Swarm (Rede de inteligência distribuída conforme Seção 3.2 do White Paper).
