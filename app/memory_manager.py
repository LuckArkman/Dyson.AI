from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from database_manager import get_db_connection

class ZeroRAMMessageHistory(BaseChatMessageHistory):
    """
    Implementação de histórico de mensagens usando a tabela sqlite conv_history.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Recupera todas as mensagens da sessão do SQLite."""
        messages = []
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, message FROM conv_history WHERE session_id = ? ORDER BY timestamp ASC",
                (self.session_id,)
            )
            rows = cursor.fetchall()
            for role, text in rows:
                if role == 'human':
                    messages.append(HumanMessage(content=text))
                else:
                    messages.append(AIMessage(content=text))
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Adiciona uma nova mensagem ao SQLite e aplica FIFO se necessário."""
        role = 'human' if isinstance(message, HumanMessage) else 'ai'
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conv_history (session_id, role, message) VALUES (?, ?, ?)",
                (self.session_id, role, message.content)
            )
            
            # FIFO: Manter apenas os últimos 10 turnos (Sprint 22 Requirement)
            cursor.execute(
                "DELETE FROM conv_history WHERE session_id = ? AND id NOT IN ("
                "SELECT id FROM conv_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT 20)",
                (self.session_id, self.session_id)
            )
            conn.commit()

    def clear(self) -> None:
        """Limpa o histórico da sessão."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conv_history WHERE session_id = ?", (self.session_id,))
            conn.commit()
