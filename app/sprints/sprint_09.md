# Sprint 09: Cálculo de Gradientes (Backward) - Etapa Inicial [CONCLUÍDA]

## Objetivos
- [x] Implementar o cálculo dos gradientes retroativos para as camadas de saída.
- [x] Desenvolver rotina de Desserialização do cálculo Forward para o cálculo Backward.

## Ferramentas & Pacotes
- [x] **NumPy (Transpose Operations)**: Transposta de pesos para Retropropagação.
- [x] **Disk Buffers**: Acesso sequencial aos arquivos Forward salvos no disco.

## Funções e Implementações
- [x] `compute_output_gradient(predicted, actual)`: Derivada da função de perda.
- [x] `backward_layer_step(grad, weights, forward_input)`: Retropropagação por camada.
- [x] `store_tensor_disk(name, grad_tensor)`: Salvar gradientes em arquivos `.npy` na pasta `grads`.

## Detalhes Técnicos
O ZeroRAM-GEN exige que, durante o Backward, o modelo busque no disco os estados que guardou no Forward (Ativações). Isso é um desafio de I/O.
