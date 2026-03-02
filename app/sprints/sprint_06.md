# Sprint 06: Mecanismo de Loading On-Demand (Forward) [CONCLUÍDA]

## Objetivos
- [x] Implementar o carregamento dinâmico de fatias de tensores (Slicing) durante a passagem para frente (Forward).
- [x] Desenvolver rotina de "liberação de RAM" após a operação para manter a pegada de memória em zero.

## Ferramentas & Pacotes
- [x] **NumPy (mmap mode)**: Carregar arquivos como se estivessem em RAM através de Memory-mapped.
- [x] **Garbage Collector (gc)**: Forçar a limpeza após cálculos de gradiente.

## Funções e Implementações
- [x] `load_tensor_mmap(layer_name)`: Carrega apenas metadados iniciais.
- [x] `dispose_tensor(obj)`: Garante `del` e `gc.collect()`.

## Detalhes Técnicos
Esta sprint é o coração do "Zero RAM". A latência de leitura por mmap deve ser monitorada.
