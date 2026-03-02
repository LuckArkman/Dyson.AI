# Sprint 27: Sistema de Cache de Vocabulário (Hot-Tokens)

## Objetivos
- Implementar cache em RAM para os tokens mais frequentes do vocabulário SQLite.
- Reduzir o número de consultas ao SQLite para palavras comuns ("o", "de", "que").

## Ferramentas & Pacotes
- **Python (dict/collections.Counter)**: Para monitoramento de frequência de uso.
- **SQLite3 (Stats)**: Tabela `VOCAB_STATS`.

## Funções e Implementações
- `build_hot_token_cache(size)`: Carregar os tokens mais frequentes do banco na memória RAM.
- `cache_lookup_with_db_fallback(text)`: Buscar ID no dicionário RAM antes do SQL.
- `update_vocab_usage(token_id)`: Incrementar contador de uso no banco SQLite.

## Detalhes Técnicos
Esta sprint é uma exceção ao "Zero RAM" (Uso de cache controlado em KB/MB) para ganho de performance em I/O.
