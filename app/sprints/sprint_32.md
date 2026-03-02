# Sprint 32: Quantização de Ativações e Gradientes

## Objetivos
- Implementar a quantização de ativações intermediárias salvas em disco.
- Reduzir o I/O do processo de Retropropagação (Backward).

## Ferramentas & Pacotes
- **NumPy (DType)**: `int8` / `uint8`.
- **Checkpointing (Disco)**: Salvar `forward_activations` como INT8.

## Funções e Implementações
- `quantize_activations(grad_block)`: Compactar dados intermediários antes da escrita em disco.
- `dequantize_activations(block)`: Reverter precisão para cálculos de gradiente Backward.
- `compare_loss_vs_precision()`: Medição estatística do impacto da quantização no erro.

## Detalhes Técnicos
A quantização de ativações pode ser dinâmica (Per-layer) para preservar a precisão onde necessário.
