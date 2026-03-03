import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from database_manager import get_db_connection
import os

def run_performance_benchmarks():
    print("ZeroRAM-GEN: Gerando Benchmarks de Performance (Sprint 44)...")
    
    with get_db_connection() as conn:
        # 1. Coletar dados de latência de I/O
        df = pd.read_sql_query("SELECT metric_name, value, context, timestamp FROM telemetry", conn)
    
    if df.empty:
        print("[!] Telemetria vazia. Rode alguns testes antes.")
        return

    # Limpeza e preparação
    df['latency_ms'] = df['value'] * 1000
    
    # Plot 1: Latência de I/O por Métrica
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='metric_name', y='latency_ms', palette="viridis")
    plt.title("ZeroRAM-GEN: Latência de I/O por Operação (ms)")
    plt.ylabel("Latência (ms)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = "d:\\Dyson.AI\\app\\performance_io.png"
    plt.savefig(plot_path)
    print(f"[OK] Gráfico de latência salvo em: {plot_path}")

    # Plot 2: Distribuição de Latência (Histograma)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['latency_ms'], bins=20, kde=True, color="blue")
    plt.title("Dyson Network: Distribuição de Latência de Busca")
    plt.xlabel("Tempo (ms)")
    
    dist_path = "d:\\Dyson.AI\\app\\latency_distribution.png"
    plt.savefig(dist_path)
    print(f"[OK] Gráfico de distribuição salvo em: {dist_path}")

if __name__ == "__main__":
    run_performance_benchmarks()
