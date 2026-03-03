# Sprint 14: Monitoramento de Treinamento e Telemetria [CONCLUÍDA]

## Objetivos
- [x] Sistema de Log para monitorar a Loss, Gradientes Norm e Tempo de I/O em tempo real.
- [x] Desenvolver rotina de alertas para degradação de performance no disco.

## Ferramentas & Pacotes
- [x] **Python (time, psutil)**: Monitor de latência de leitura/escrita e uso de RAM.
- [x] **SQLite3 (Metrics)**: Tabela `TELEMETRY`.

## Funções e Implementações
- [x] `log_telemetry(metric_name, value)`: Grava métricas no baco de dados.
- [x] `save_tensor_logged()` / `load_tensor_mmap()`: Medição automática de latência de I/O.
- [x] Integração da telemetria no loop principal em `trainer.py`.

## Detalhes Técnicos
Os logs revelaram uma latência de escrita de ~1.2s para os pesos maiores, o que confirma o disco como o principal gargalo. O pico de RAM de ~290MB está dentro do esperado para o processamento de fatias e estados do Adam.
