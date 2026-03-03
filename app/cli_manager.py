import argparse
import sys
import time
from engine import dense_layer_forward, embedding_lookup
from database_manager import init_db, get_db_connection
from tensor_manager import ensure_v0_weights
from network_manager import ZeroRAMNetworkManager

def main_cli():
    parser = argparse.ArgumentParser(description="ZeroRAM-GEN CLI - Comando e Controle")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Comando: Status
    subparsers.add_parser("status", help="Mostra o status do motor e do disco")

    # Comando: Net (Swarm)
    net_parser = subparsers.add_parser("net", help="Gerencia a rede Swarm")
    net_parser.add_argument("--register", action="store_true", help="Registra o nó local")
    net_parser.add_argument("--list", action="store_true", help="Lista nós no Swarm")

    # Comando: Telemetry
    subparsers.add_parser("stats", help="Mostra estatísticas de I/O em tempo real")

    args = parser.parse_args()

    if args.command == "status":
        print("--- ZeroRAM-GEN Status ---")
        init_db()
        ensure_v0_weights()
        print("[OK] Banco de dados e Pesos verificados.")
        
    elif args.command == "net":
        nm = ZeroRAMNetworkManager()
        if args.register:
            nm.register_node()
        if args.list:
            nodes = nm.get_available_nodes()
            print(f"Nós ativos no Swarm: {nodes}")

    elif args.command == "stats":
        print("Estatísticas de Telemetria (Últimos 10 registros de I/O):")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metric_name, value, context, timestamp FROM telemetry ORDER BY timestamp DESC LIMIT 10")
            rows = cursor.fetchall()
            for row in rows:
                print(f"[{row[3]}] {row[0]}: {row[1]:.6f}s | Context: {row[2]}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()
