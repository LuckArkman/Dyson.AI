# Sprint 01: Setup do Ambiente e Base de Dados de Vocabulário

## Objetivos
- Estabelecer o ambiente de desenvolvimento.
- Validar a estrutura do banco SQLite para o vocabulário.
- Implementar as primeiras funções de conexão e gerenciamento de banco.

## Ferramentas & Pacotes
- **Python 3.10+**: Linguagem principal.
- **SQLite3**: Driver nativo para persistência de vocabulário.
- **Venv/Conda**: Isolamento de dependências.

## Funções e Implementações
- `get_db_connection()`: Estabelecer conexão persistente com `vocab.db`.
- `init_db()`: Criar tabela `VOCAB` caso não exista, conforme especificado no White Paper.
- `close_db()`: Garantir o fechamento seguro das conexões.

## Detalhes Técnicos
A tabela `VOCAB` deve possuir os campos `ID` (Primary Key) e `TEXT` (Varchar 255). Esta base é a âncora para todas as operações de tokenização subsequentes.
