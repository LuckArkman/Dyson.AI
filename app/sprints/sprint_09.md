# Sprint 09: Cálculo de Gradientes (Backward) - Etapa Inicial

## Objetivos
- Implementar o cálculo dos gradientes retroativos para as camadas de saída.
- Desenvolver rotina de Desserialização do cálculo Forward para o cálculo Backward.

## Ferramentas & Pacotes
- **NumPy (Transpose Operations)**: Transposta de pesos para Retropropagação.
- **Disk Buffers**: Acesso sequencial aos arquivos Forward salvos no disco.

## Funções e Implementações
- `compute_output_gradient(predicted, actual)`: Derivada da função de perda.
- `backward_layer_step(grad, weights, forward_input)`: Retropropagação por camada.
- `store_gradients_on_disk(name, grad_tensor)`: Salvar gradientes em arquivos `.grad`.

## Detalhes Técnicos
O ZeroRAM-GEN exige que, durante o Backward, o modelo busque no disco os estados que guardou no Forward (Ativações). Isso é um desafio de I/O.
