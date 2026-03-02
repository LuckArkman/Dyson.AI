# Sprint 07: Motor de Operações Matemáticas (Forward) [CONCLUÍDA]

## Objetivos
- [x] Implementar as funções de ativação e o motor de propagação das camadas iniciais.
- [x] Desenvolver a camada de Embeddings (Mapeamento ID -> Tensor).

## Ferramentas & Pacotes
- [x] **NumPy**: Operações de Álgebra Linear.
- [x] **ReLU/Sigmoid/ReLU6**: Funções de ativação para as camadas.

## Funções e Implementações
- [x] `embedding_lookup(id)`: Mapear ID (do SQLite) para o vetor de embedding (do disco).
- [x] `apply_activation(tensor, type)`: Modularizar a função de ativação.
- [x] `forward_propagation_step(layer_index, input_tensor)`: Integra o carregamento e cálculo.

## Detalhes Técnicos
Cada passo do Forward deve serializar o estado intermediário em um arquivo temporário no disco, caso a RAM esteja em limite crítico.
