# Sprint 10: Retropropagação (Backward) - Camadas de Hidden e Embedding [CONCLUÍDA]

## Objetivos
- [x] Concluir o cálculo de gradientes para as camadas internas e para a matriz de Embeddings.
- [x] Implementar o fluxo de erros de volta até os pesos iniciais.

## Ferramentas & Pacotes
- [x] **NumPy**: Para manipulação das fatias de tensores escondidos.
- [x] **SQLite3 (Logging)**: Registro de magnitude de gradientes (Para evitar explosão/desaparecimento).

## Funções e Implementações
- [x] `backward_layer_step(grad_out, weights_name, forward_input)`: Retropropagação aprimorada com bias.
- [x] `accumulate_embedding_grad(ids, grad_item)`: Calcular o gradiente esparso para o vocabulário.
- [x] `reset_accumulated_grads()`: Rotina para gerenciar o acúmulo de gradientes em disco.

## Detalhes Técnicos
Esta sprint finaliza o motor matemático do modelo em modo Zero RAM (carrega, calcula, salva, limpa).
