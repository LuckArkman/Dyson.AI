# Sprint 14: Otimização de I/O - Pré-fetch de Tensores

## Objetivos
- Implementar o carregamento antecipado (Prefetch) de pesos para a próxima camada enquanto a atual está em cálculo.
- Reduzir o tempo de espera do CPU durante o acesso a disco.

## Ferramentas & Pacotes
- **Python (threading/asyncio)**: Para carregar pesos em paralelo ao cálculo Forward.
- **NumPy (Shared Memory)**: Buffer de antecipação.

## Funções e Implementações
- `prefetch_next_layer_weights(layer_index)`: Inicia leitura do arquivo de pesos em thread separada.
- `async_save_weights(data, path)`: Salvar pesos sem bloquear o loop principal.
- `wait_for_io_ready(target_path)`: Sincronização de threads.

## Detalhes Técnicos
Esta sprint visa mitigar a "Desvantagem de Latência de I/O" citada no White Paper. Minimiza a ociosidade do motor matemático.
