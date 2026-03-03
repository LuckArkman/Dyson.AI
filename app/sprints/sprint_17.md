# Sprint 17: Interface de Chat CLI e Loop de Conversa [CONCLUÍDA]

## Objetivos
- [x] Implementar um script de chat interativo (CLI).
- [x] Desenvolver mecanismo de "streaming" simulado (exibir token a token).
- [x] Suporte a comandos especiais (ex: /exit, /clear).

## Ferramentas & Pacotes
- [x] **Python (input/sys)**: Loop de leitura e escrita no terminal.
- [x] **inference.py**: Motor de geração de texto integrado.

## Funções e Implementações
- [x] `chat_loop()`: Loop principal de interação Usuário <-> Robô.
- [x] **Streaming de Tokens**: Exibição em tempo real conforme os tokens são preditos.
- [x] **Gestão de Contexto**: Histórico de conversa mantido para o motor de inferência.

## Detalhes Técnicos
O chat foi validado com um loop simulado, demonstrando a capacidade de receber prompts, predizer sequências e exibi-las palavra por palavra. O uso de memória permanece fixo, independente do tamanho do vocabulário, graças à arquitetura Zero RAM.
