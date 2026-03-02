# Sprint 43: Interface de Usuário Final (Terminal CLI e Web)

## Objetivos
- Implementar a CLI (Interface de Linha de Comando) interativa.
- Desenvolver um DashBoard Web simples para telemetria I/O.

## Ferramentas & Pacotes
- **Python (Click / Typer)**: Para a interface CLI.
- **Python (Flask / FastAPI)**: Para o dashboard de telemetria.

## Funções e Implementações
- `zeroram_cli(...)`: Interface de comando Principal do projeto.
- `start_telemetry_ui()`: Lançar servidor HTTP com visualização de métricas de disco.
- `connect_cli_to_swarm()`: Modo CLI para nó do Swarm.

## Detalhes Técnicos
Transforme o motor de disco em um produto interativo. O dashboard deve mostrar o "Gargalo de I/O" em gráficos.
