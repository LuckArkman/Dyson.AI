# Sprint 31: Quantização de Pesos - Etapa 1 (Conversão para INT8)

## Objetivos
- Implementar a redução de precisão dos pesos de 32 bits (FP32) para 8 bits (INT8).
- Reduzir o tamanho dos arquivos de pesos em disco em 75%.

## Ferramentas & Pacotes
- **NumPy (DType)**: `int8` para armazenamento compactado em disco.
- **Python (Min-Max Scaling)**: Funções de normalização para o range [-128, 127].

## Funções e Implementações
- `quantize_fp32_to_int8(tensor)`: Mapear pesos contínuos para a grade de inteiros.
- `store_quantized_weights(path, weights, scale, zero_point)`: Salvar no disco com metadados de restauração.
- `dequantize_on_demand(int_weights, scale, zero_point)`: Restaurar para cálculo em RAM.

## Detalhes Técnicos
A quantização impacta diretamente na acuracidade do modelo, mas é vital para o Roadmap (Fase 3) do White Paper.
