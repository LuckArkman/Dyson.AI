# Sprint 22: Gerenciamento de Memória Contextual (Conversacional)

## Objetivos
- Implementar a janela de contexto (Context Window) para manter o histórico da conversa.
- Desenvolver a rotina de limpeza de contexto antigo (FIFO).

## Ferramentas & Pacotes
- **LangChain (Memory)**: Gestão do histórico de mensagens.
- **SQLite3 (History Storage)**: Tabela `CONV_HISTORY`.

## Funções e Implementações
- `update_context_buffer(msg)`: Adicionar nova mensagem ao histórico do modelo.
- `truncate_history(max_tokens)`: Remover mensagens antigas para manter o limite de tokens dentro do orçamento de RAM/Disco.
- `get_formatted_context()`: Concatenar o contexto como prefixo para o próximo passo de inferência.

## Detalhes Técnicos
O ZeroRAM-GEN precisa carregar o histórico do contexto no embed_lookup. O custo de I/O aumenta com o tamanho do contexto.
