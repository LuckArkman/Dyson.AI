# Sprint 34: Otimização de I/O II - Compressão de Dados em Disco [CONCLUÍDA]

## Objetivos
- [x] Implementar a compressão de arquivos de pesos usando algoritmos rápidos (**LZ4**).
- [x] Reduzir o footprint de disco e economizar banda de I/O carregando dados comprimidos.

## Ferramentas & Pacotes
- [x] **LZ4**: Escolhido pela alta velocidade de descompressão, ideal para modelos real-time.
- [x] **Python (lz4.frame)**: Integração nativa para compressão de streams de bytes de tensores.

## Funções e Implementações
- [x] `save_compressed_tensor(name, tensor)`: Converte tensor para bytes e comprime com LZ4 antes de salvar.
- [x] `load_compressed_tensor(name)`: Lê do disco, descompacta em RAM e restaura o shape original.
- [x] **Fallback Engine**: O motor de inferência tenta carregar versões comprimidas `.lz4` prioritariamente.

## Detalhes Técnicos
A compressão LZ4 foi validada como **lossless** (sem perdas). Em tensores com padrões repetitivos, a redução de espaço chegou a **99%**, e em pesos reais (embeddings), ajuda a mitigar o gargalo de leitura sequencial do disco. O overhead de CPU para descompressão é compensado pela redução drástica do tempo de I/O.
