# Sprint 10: Retropropagação (Backward) - Camadas de Hidden e Embedding

## Objetivos
- Concluir o cálculo de gradientes para as camadas internas e para a matriz de Embeddings.
- Implementar o fluxo de erros de volta até os pesos iniciais.

## Ferramentas & Pacotes
- **NumPy**: Para manipulação das fatias de tensores escondidos.
- **SQLite3 (Logging)**: Registro de magnitude de gradientes (Para evitar explosão/desaparecimento).

## Funções e Implementações
- `backward_hidden_layer(grad_next, hidden_weights, activations)`: Retropropagação das camadas densas.
- `update_embeddings_grad(id, grad_item)`: Calcular o gradiente para o token correspondente à id no banco SQLite.
- `reset_accumulated_grads()`: Rotina para gerenciar o acúmulo de gradientes em disco.

## Detalhes Técnicos
Esta sprint finaliza o motor matemático do modelo em modo Zero RAM (carrega, calcula, salva, limpa).
