# Sprint 06: Mecanismo de Loading On-Demand (Forward)

## Objetivos
- Implementar o carregamento dinâmico de fatias de tensores (Slicing) durante a passagem para frente (Forward).
- Desenvolver rotina de "liberação de RAM" após a operação para manter a pegada de memória em zero.

## Ferramentas & Pacotes
- **NumPy (mmap mode)**: Carregar arquivos como se estivessem em RAM através de Memory-mapped.
- **Garbage Collector (gc)**: Forçar a limpeza após cálculos de gradiente.

## Funções e Implementações
- `load_tensor_slice(layer_name, slice_coords)`: Apenas as fatias necessárias para a operação atual.
- `exec_forward_operation(input_data, layer_weights)`: Operação de produto escalar (dot product).
- `dispose_layer_memory(obj)`: Garante `del` e `gc.collect()`.

## Detalhes Técnicos
Esta sprint é o coração do "Zero RAM". A latência de leitura por mmap deve ser monitorada.
