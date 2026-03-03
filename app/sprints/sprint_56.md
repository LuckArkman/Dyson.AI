# Sprint 56: Quantização Extrema (INT4) e Compressão de Vocabulário

## Objetivos
- [ ] Implementar suporte para Pesos INT4 para reduzir o tamanho dos arquivos de 53k tokens.
- [ ] Desenvolver o motor de dequantização sub-8-bit em tempo real (Dyson Bits).
- [ ] Reduzir o footprint de disco das matrizes de embedding de ~300MB para ~75MB.

## Ferramentas & Pacotes
- **Numpy (Bit Manipulation)**: Empacotamento de dois pesos de 4 bits em um único byte.
- **ZeroRAM Optimizer**: Rotinas de compressão pós-treinamento.

## Funções e Implementações
- `quantize_to_int4(tensor)`: Conversão de pesos FP32/FP16 para representação de 4 bits.
- `dequantize_from_int4(...)`: Recuperação rápida durante o Forward Pass.
- `compress_v11_pt()`: Script para processar o checkpoint PT com as novas otimizações.

## Detalhes Técnicos
Com um vocabulário de 53k tokens, as matrizes de pesos cresceram significativamente. A quantização INT4 é vital para manter o ZeroRAM-GEN eficiente em hardware de entrada, permitindo que o modelo "lusófono" caiba em dispositivos com barramentos de disco limitados sem perder a qualidade da linguagem.
