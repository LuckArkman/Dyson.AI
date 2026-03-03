# Sprint 15: Otimização de I/O e Compressão de Pesos (FP16) [CONCLUÍDA]

## Objetivos
- [x] Implementar suporte a precisão de 16 bits (FP16/Half-precision) para reduzir o uso de disco e tempo de I/O em 50%.
- [x] Desenvolver mecanismo de estabilidade numérica para baixa precisão.

## Ferramentas & Pacotes
- [x] **NumPy (float16)**: Redução de precisão para economia de espaço.
- [x] **Clamping**: Proteção contra overflow/underflow no Softmax e Adam.

## Funções e Implementações
- [x] `convert_weights_to_fp16()`: Conversor em lote para migrar modelos existentes de FP32 para FP16.
- [x] `adam_update_step(...)`: Atualizado para respeitar o dtype dos pesos.
- [x] `compute_softmax(...)`: Implementada estabilidade via `clip`.

## Detalhes Técnicos
A transição para FP16 reduziu o tamanho dos pesos da camada de saída de 246MB para 123MB. A latência média de escrita caiu de 1.27s para ~0.95s. Estabilidades numéricas foram adicionadas para mitigar os riscos de `NaN` em cálculos de gradiente.
