# Sprint 24: Embeddings Condicionais (Persona Embedding)

## Objetivos
- Implementar a camada de Embeddings condicionais para ajuste fino de comportamento em tempo de execução.
- Desenvolver rotina de "Bias" comportamental por categoria (Criativo vs. Técnico).

## Ferramentas & Pacotes
- **NumPy (Math)**: Somar vetores de embedding no disco.
- **SQLite3 (Modulation)**: Tabela `BIAS_TEMPLATES`.

## Funções e Implementações
- `add_behavioral_bias(embeddings, bias_name)`: Aplicar soma vetorial de viés ao prompt em RAM.
- `store_bias_vectors(name, vector)`: Salvar novos arquétipos de persona.
- `toggle_persona_mode(mode_id)`: Switching rápido de persona no motor Zero RAM.

## Detalhes Técnicos
Isso permite o "Sistema de Modulação Comportamental" (White Paper, Seção 8) sem precisar retreinar o modelo completo.
