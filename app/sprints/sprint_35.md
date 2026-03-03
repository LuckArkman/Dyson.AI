# Sprint 35: Segmentação de Tensores (Sharding) no Disco [CONCLUÍDA]

## Objetivos
- [x] Dividir tensores gigantes (como os Embeddings) em arquivos de fragmentos (Shards).
- [x] Reduzir o custo de I/O de busca de ID único no disco.

## Ferramentas & Pacotes
- [x] **Python (os, shutil)**: Gerenciamento de diretórios de shards.
- [x] **SQLite3 (Metadata)**: Tabela `shard_map` para rastrear intervalos de cada fragmento.

## Funções e Implementações
- [x] `create_tensor_shards(name, tensor, ids_per_shard)`: Divisão física automática de tensores.
- [x] `lookup_shard_for_id(tensor_name, original_id)`: Localização dinâmica de fragmentos via Banco de Dados.
- [x] **Multi-Shard Inference**: `embedding_lookup` atualizado para montar vetores a partir de múltiplos shards de forma transparente.

## Detalhes Técnicos
O ZeroRAM-GEN agora possui uma granularidade muito maior. Em vez de operar sobre um único arquivo de 64MB+ (que pode ser problemático em sistemas com pouca cache de disco/RAM), o motor pode ler apenas pequenos fragmentos de 5000 tokens por vez. Isso é a base para a **Escalabilidade em Swarm** (Sprint 36), onde cada nó pode segurar apenas alguns shards.
