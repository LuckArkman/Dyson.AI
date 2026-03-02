# Sprint 37: Escabilidade - Etapa 2 (Treinamento Distribuído em Disco)

## Objetivos
- Implementar a sincronização de gradientes entre discos em redes locais.
- Desenvolver rotina de "Gradient Accumulation" coletiva (All-reduce simulado em disco).

## Ferramentas & Pacotes
- **Python (Requests/Socket)**: Transferência de gradientes.
- **SQLite3 (Stats Distributed)**: Monitorar progresso coletivo.

## Funções e Implementações
- `distribute_grad_sync(node_list, grad_tensor)`: Enviar gradientes locais para agregação.
- `aggregate_swarm_gradients(grads)`: Calcular média dos gradientes coletivos no disco mestre.
- `update_swarm_weights()`: Distribuir os novos pesos para o disco de cada nó do Swarm.

## Detalhes Técnicos
Treinar um ZeroRAM-GEN em 10 máquinas aumenta a performance proporcionalmente ao paralelismo dos discos.
