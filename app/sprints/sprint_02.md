# Sprint 02: Pipeline de Tokenização - Etapa 1 (Ingestão)

## Objetivos
- Desenvolver o mecanismo de extração de tokens a partir do arquivo bruto `pt_0.txt`.
- Implementar as rotinas de limpeza e normalização de texto (Letras minúsculas, remoção de caracteres especiais).

## Ferramentas & Pacotes
- **Python Standard Library (re)**: Expressões regulares para tokenização avançada.
- **PT_0.txt**: Dataset bruto disponível em `Dayson/pt_0.txt`.

## Funções e Implementações
- `load_raw_data(path)`: Carregar o arquivo original de forma eficiente (Line-by-line para evitar estouro de memória).
- `normalize_text(text)`: Limpeza de strings para garantir unicidade no vocabulário.
- `tokenize_line(line)`: Divisão de frases em tokens unitários (Words/Subwords).

## Detalhes Técnicos
O foco desta sprint é a eficiência de leitura, preparando os dados para a indexação no SQLite. Deve ser compatível com o dataset de ~600.000 tokens.
