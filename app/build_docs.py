import inspect
import engine
import tensor_manager
import database_manager
import os

def build_api_docs():
    """Gera um resumo da documentação técnica (API Reference) das Sprints anteriores."""
    print("ZeroRAM-GEN: Gerando Documentação Técnica (Sprint 41)...")
    
    docs_content = "# ZeroRAM-GEN: Documentação Técnica da API\n\n"
    docs_content += "Este documento descreve as principais funções do motor de disco, quantização e rede.\n\n"
    
    modules = {
        "Motor de Inferência (engine.py)": engine,
        "Gerenciador de Tensores (tensor_manager.py)": tensor_manager,
        "Banco de Dados & Vocabulário (database_manager.py)": database_manager
    }
    
    for mod_name, mod in modules.items():
        docs_content += f"## {mod_name}\n"
        functions = [f for f in inspect.getmembers(mod, inspect.isfunction)]
        for name, func in functions:
            if func.__module__ == mod.__name__:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or "Sem descrição."
                docs_content += f"### `{name}{sig}`\n{doc}\n\n"
    
    output_path = "d:\\Dyson.AI\\app\\API_REFERENCE.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(docs_content)
    
    print(f"[OK] Documentação gerada em: {output_path}")

if __name__ == "__main__":
    build_api_docs()
