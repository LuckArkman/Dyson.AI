from flask import Flask, render_template_string, jsonify
from database_manager import get_db_connection

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ZeroRAM-GEN Dashboard</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333; }
        h1 { color: #00e5ff; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #333; }
        th { color: #00e5ff; }
        .metric { font-size: 1.2em; font-weight: bold; color: #76ff03; }
    </style>
</head>
<body>
    <h1>Dyson Network: ZeroRAM-GEN Dashboard</h1>
    
    <div class="card">
        <h2>Últimas Métricas de Telemetria (I/O)</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Métrica</th>
                <th>Valor (sec)</th>
                <th>Contexto</th>
            </tr>
            {% for row in telemetry %}
            <tr>
                <td>{{ row[3] }}</td>
                <td>{{ row[0] }}</td>
                <td class="metric">{{ "%.6f"|format(row[1]) }}</td>
                <td>{{ row[2] }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h2>Progresso do Treinamento (Convergência)</h2>
        <table>
            <tr>
                <th>Época</th>
                <th>Step</th>
                <th>Loss</th>
                <th>Timestamp</th>
            </tr>
            {% for row in train_logs %}
            <tr>
                <td>{{ row[0] }}</td>
                <td>{{ row[1] }}</td>
                <td class="metric" style="color: #ff5252;">{{ "%.6f"|format(row[2]) }}</td>
                <td>{{ row[3] }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h2>Nós do Swarm</h2>
        <ul>
            {% for node in nodes %}
            <li><strong>{{ node[0] }}</strong> - {{ node[1] }} ({{ node[2] }})</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Telemetria I/O
        cursor.execute("SELECT metric_name, value, context, timestamp FROM telemetry ORDER BY timestamp DESC LIMIT 20")
        telemetry = cursor.fetchall()
        
        # Histórico de Treino
        cursor.execute("SELECT epoch, step, loss, timestamp FROM train_log ORDER BY timestamp DESC LIMIT 15")
        train_logs = cursor.fetchall()
        
        # Swarm Nodes
        cursor.execute("SELECT node_id, base_url, status FROM network_nodes")
        nodes = cursor.fetchall()
        
    return render_template_string(HTML_TEMPLATE, telemetry=telemetry, nodes=nodes, train_logs=train_logs)

if __name__ == "__main__":
    print("Iniciando Dashboard de Treino em http://localhost:5000")
    app.run(port=5000)
