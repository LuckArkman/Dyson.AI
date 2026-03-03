# Sprint 22: Gerenciamento de Memória Contextual (Conversacional) [CONCLUÍDA]

## Objetivos
- [x] Implementar a janela de contexto (Context Window) para manter o histórico da conversa.
- [x] Desenvolver a rotina de limpeza de contexto antigo (FIFO).

## Ferramentas & Pacotes
- [x] **LangChain (Memory)**: Gestão do histórico de mensagens integrada.
- [x] **SQLite3 (History Storage)**: Tabela `conv_history` para persistência de longo prazo.

## Funções e Implementações
- [x] `ZeroRAMMessageHistory`: Classe de histórico customizada para SQLite.
- [x] **Otimização de Descarte**: Melhoria no `dispose_tensor` para acelerar ciclos de inferência recursiva.
- [x] **Context Window**: Histórico recuperado e injetado via LangChain `MessagesPlaceholder`.

## Detalhes Técnicos
O modelo agora consegue "lembrar" de interações passadas na mesma sessão. A persistência em SQLite garante que o contexto não seja perdido entre execuções, e a otimização do GC permitiu que a geração com histórico longo permanecesse viável na arquitetura de disco.
