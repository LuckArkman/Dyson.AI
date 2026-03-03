# Sprint 31: Quantização de Pesos - Etapa 1 (Conversão para INT8) [CONCLUÍDA]

## Objetivos
- [x] Implementar a redução de precisão dos pesos de 32 bits (FP32) para 8 bits (INT8).
- [x] Reduzir o tamanho dos arquivos de pesos em disco em 75% (comparado ao FP32 original).

## Ferramentas & Pacotes
- [x] **NumPy (DType)**: Uso de `int8` para armazenamento compactado e `mmap` para leitura.
- [x] **Python (Quantization Logic)**: Cálculo dinâmico de `scale` e `zero_point` por camada.

## Funções e Implementações
- [x] `quantize_to_int8(tensor)`: Conversão linear para o range inteiro [-128, 127].
- [x] `save_quantized_tensor()`: Persistência de pesos e metadados (.meta) no disco.
- [x] `dequantize_from_int8()`: Restauração transparente para float durante o Forward.
- [x] `convert_weights_to_int8()`: Utilitário para migração total do modelo.

## Detalhes Técnicos
A quantização INT8 foi validada com sucesso. O motor Zero RAM agora carrega tensores de 8 bits do disco, dequantizando apenas os fragmentos necessários em RAM para a computação. Isso reduz significativamente o I/O de disco mantendo a integridade semântica básica do modelo.
