# Sprint 14: Monitoramento de Treinamento e Telemetria

## Objetivos
- Sistema de Log para monitorar a Loss, Gradientes Norm e Tempo de I/O em tempo real.
- Desenvolver rotina de alertas para degradação de performance no disco.

## Ferramentas & Pacotes
- **Python (time, psutil)**: Monitor de latência de leitura/escrita e uso de RAM.
- **SQLite3 (Metrics)**: Tabela `TELEMETRY`.

## Funções e Implementações
- `measure_io_latency(fn, *args)`: Decorador para medir tempo das funções de carregamento/salvamento de pesos.
- `calculate_perf_stats()`: Resumo de tokens por segundo (TPS) e tempo de treinamento esperado.
- `early_stopping_check(loss_delta)`: Critério para pausa automática de treino.

## Detalhes Técnicos
Devido ao custo de I/O inerente ao ZeroRAM, estatísticas de tempo de disco são vitais para futuras otimizações de cache.
