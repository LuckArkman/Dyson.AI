# Sprint 34: Otimização de I/O II - Compressão de Dados em Disco

## Objetivos
- Implementar a compressão de arquivos de pesos usando algoritmos rápidos (LZ4 ou Zstandard).
- Reduzir o footprint inicial de tokens do dataset `pt_0.txt`.

## Ferramentas & Pacotes
- **LZ4 / Zstandard**: Para compressão/descompressão rápida de blocos de dados.
- **FileSystem (OS)**: Monitoramento de taxa de transferência de disco compactado.

## Funções e Implementações
- `compress_weight_file(path)`: Comprimir arquivos de pesos no fechamento do treinamento.
- `decompress_chunk_on_demand(compressed_buffer)`: Restaurar em RAM apenas o que for operar agora.
- `measure_overhead_compression()`: Balancear tempo de descompressão vs. tempo de leitura pura.

## Detalhes Técnicos
O ZeroRAM-GEN usa a CPU extra para descomprimir, poupando a largura de banda do disco rígido.
