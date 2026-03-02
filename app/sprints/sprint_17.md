# Sprint 17: Motor de Inferência - Etapa 2 (Passagem Forward de Inferência)

## Objetivos
- Executar o cálculo Forward do modelo com os pesos treinados no MVP.
- Garantir que a inferência não altere os pesos originais do modelo.

## Ferramentas & Pacotes
- **NumPy (Mode ReadOnly)**: Carregamento de pesos do disco em modo somente-leitura.
- **Cache de Ativações (Disk-Stored)**: Persistência temporária de estados internos de inferência.

## Funções e Implementações
- `inference_forward_pass(prompt_tokens)`: Cálculo do estado final da rede sem treinamento.
- `prevent_weight_overwrite()`: Garantir que pesos não sofram alteração durante inferência.
- `extract_last_logits()`: Mapear o último estado para probabilidades de próximo token.

## Detalhes Técnicos
A inferência economiza o cálculo de Backward, mas mantém o custo de I/O de carregamento de pesos Forward por camada.
