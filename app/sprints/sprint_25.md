# Sprint 25: Orquestração Avançada (Chain-of-Thought) via LangChain

## Objetivos
- Implementar o pensamento passo-a-passo (CoT) através da LangChain.
- Desenvolver rotina de "Pensamento Intermediário" (Internal Thoughts).

## Ferramentas & Pacotes
- **LangChain (LLMChain)**: Para encadeamento de prompts internos e externos.
- **Python (Parsing)**: Extração de raciocínio da saída bruta.

## Funções e Implementações
- `create_cot_prompt(task)`: Solicitar que o ZeroRAM-GEN pense antes da resposta.
- `execute_reasoning_loop(prompt)`: Loop que resolve a tarefa em sub-problemas.
- `parse_final_answer(raw_text)`: Limpeza de texto de pensamento para resposta ao usuário.

## Detalhes Técnicos
Devido à baixa latência, o CoT deve ser econômico em chamadas ao disco. Cada passo de CoT é um novo ciclo de Forward.
