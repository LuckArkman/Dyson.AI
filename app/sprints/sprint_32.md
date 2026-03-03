# Sprint 32: Quantização de Ativações e Gradientes [CONCLUÍDA]

## Objetivos
- [x] Implementar a quantização de ativações intermediárias salvas em disco.
- [x] Reduzir o I/O do processo de Retropropagação (Backward).

## Ferramentas & Pacotes
- [x] **NumPy (INT8)**: Compactação de tensores temporários para o range [-128, 127].
- [x] **Checkpointing**: Integração com `store_tensor_disk` para suporte a flag `quantize`.

## Funções e Implementações
- [x] `store_tensor_disk(..., quantize=True)`: Novo parâmetro para ativar a compactação dinâmica.
- [x] `load_tensor_disk(...)`: Detecção automática de versões quantizadas (_q.npy) e restauração transparente.

## Detalhes Técnicos
A quantização de ativações reduziu o consumo de disco em aproximadamente 72% por tensor temporário, mantendo um MSE extremamente baixo (0.00007). Isso é crucial para o treinamento "Zero RAM", onde o gargalo principal é a velocidade de escrita das ativações no disco durante o Forward para uso posterior no Backward.
