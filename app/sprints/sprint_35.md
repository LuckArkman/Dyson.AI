# Sprint 35: Segmentação de Tensores (Sharding) no Disco

## Objetivos
- Dividir tensores gigantes (como os Embeddings) em arquivos de fragmentos (Shards).
- Reduzir o custo de I/O de busca de ID único no disco.

## Ferramentas & Pacotes
- **Python (os, shutil)**: Gerenciamento de diretórios de shards.
- **SQLite3 (Metadata)**: Tabela `SHARD_MAP`.

## Funções e Implementações
- `create_embedding_shards(total_ids, ids_per_shard)`: Dividir a matriz de embedding.
- `lookup_shard_by_id(token_id)`: Localizar em qual fragmento de arquivo está o vetor desejado.
- `load_specific_shard(shard_id)`: Carregar apenas o fragmento necessário em vez do arquivo total.

## Detalhes Técnicos
O ZeroRAM-GEN ganha em granularidade, evitando o carregamento de "lixo" do disco ao buscar um token simples.
