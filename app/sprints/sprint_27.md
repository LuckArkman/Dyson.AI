# Sprint 27: Sistema de Cache de Vocabulário (Hot-Tokens) [CONCLUÍDA]

## Objetivos
- [x] Implementar cache em RAM para os tokens mais frequentes do vocabulário SQLite.
- [x] Reduzir o número de consultas ao SQLite para palavras comuns ("o", "de", "que").

## Ferramentas & Pacotes
- [x] **Python (dict)**: Cache bidirecional (ID->Text e Text->ID) para lookup instantâneo.
- [x] **SQLite3 (Stats)**: Tabela `vocab_stats` para rastrear a frequência de uso dos tokens.

## Funções e Implementações
- [x] `update_vocab_usage(token_id)`: Registra a frequência de uso no banco.
- [x] `build_hot_token_cache(size)`: Pré-carrega os tokens mais comuns na inicialização.
- [x] **Lookup Bidirecional**: Otimização dos métodos de busca para priorizar a RAM.

## Detalhes Técnicos
A inclusão do cache de Hot-Tokens reduziu a latência de lookup para quase zero (0.0001s para 1000 consultas). Embora o projeto seja Zero RAM para tensores, o uso de alguns KB para o vocabulário frequente é uma otimização estratégica que remove o gargalo de I/O do banco de dados durante a tokenização.
