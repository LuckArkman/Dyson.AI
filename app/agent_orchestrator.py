from typing import List, Union, Any
from langchain_core.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_wrapper import ZeroRAMLLM
from langchain_core.prompts import PromptTemplate
import re

# 1. Definir Ferramentas
def get_time(query):
    import datetime
    return str(datetime.datetime.now())

tools = [
    Tool(
        name="GetTime",
        func=get_time,
        description="Útil para quando você precisa saber a hora atual."
    )
]

# 2. Template ReAct
template = """Você é um assistente que pode usar ferramentas.
Suas ferramentas são:
{tools_desc}

Use o seguinte formato:
Pergunta: {input}
Pensamento: você deve sempre pensar sobre o que fazer
Ação: a ação a ser tomada (uma de [{tool_names}])
Entrada da Ação: a entrada para a ação
Observação: o resultado da ação...
Resposta Final: a resposta final

Comece!
Pergunta: {input}
Pensamento: """

class ZeroRAMAgentOrchestrator:
    def __init__(self, llm: ZeroRAMLLM, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.tool_names = ", ".join(self.tools.keys())
        self.tools_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])

    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Resposta Final:" in text:
            return AgentFinish(return_values={"output": text.split("Resposta Final:")[-1].strip()}, log=text)
        
        regex = r"Ação: (.*?)[\n]*Entrada da Ação:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            return AgentFinish(return_values={"output": text}, log=text)
            
        return AgentAction(tool=match.group(1).strip(), tool_input=match.group(2).strip(), log=text)

    def run(self, input_text: str):
        print(f"\n[AGENT] Iniciando tarefa: {input_text}")
        
        prompt = template.format(
            input=input_text,
            tools_desc=self.tools_desc,
            tool_names=self.tool_names
        )
        
        # Loop ReAct simplificado (max 3 passos)
        for i in range(3):
            print(f"--- Passo {i+1} ---")
            output = self.llm.invoke(prompt)
            print(f"Log do Modelo: {output}")
            
            result = self._parse_output(output)
            
            if isinstance(result, AgentFinish):
                return result.return_values["output"]
            
            # Executar Ação
            tool_name = result.tool
            if tool_name in self.tools:
                observation = self.tools[tool_name].run(result.tool_input)
                print(f"Observação: {observation}")
                prompt += f"{output}\nObservação: {observation}\nPensamento: "
            else:
                return f"Erro: Ferramenta {tool_name} não encontrada."
                
        return "Erro: Limite de passos atingido."
