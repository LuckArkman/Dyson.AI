# Sprint 39: Handover de Inferência (Transferência de Responsabilidade) [CONCLUÍDA]

## Objetivos
- [x] Implementar a lógica de Handover de inferência baseada em carga.
- [x] Eleger nós com menor latência de I/O para processar camadas específicas.

## Ferramentas & Pacotes
- [x] **psutil**: Medição de recursos locais (CPU/RAM).
- [x] **Telemetry API**: Uso histórico de latência de disco para decisão de roteamento.

## Funções e Implementações
- [x] `measure_node_load()`: Score dinâmico de disponibilidade do nó.

## Detalhes Técnicos
O Swarm agora é inteligente: camadas com maior custo de I/O são migradas para nós com discos mais rápidos ou menos ocupados.
