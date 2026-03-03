from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from inference import generate_text

class ZeroRAMLLM(LLM):
    """
    Wrapper customizado para o ZeroRAM-GEN compatível com LangChain.
    """
    
    max_new_tokens: int = 20
    temperature: float = 0.7
    top_k: int = 40

    @property
    def _llm_type(self) -> str:
        return "zeroram_gen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Executa a geração de texto usando o motor Zero RAM.
        """
        # A interface generate_text já lida com o motor
        response = generate_text(
            prompt, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            stop_on_punctuation=True
        )
        
        # O retorno do LangChain deve ser apenas o texto gerado DEPOIS do prompt
        # Nossa função generate_text retorna [prompt + gerado]
        # Vamos tentar extrair apenas a parte nova
        if response.startswith(prompt.lower()):
            response = response[len(prompt):].strip()
            
        return response

    @property
    def _identifying_params(self) -> dict:
        """Parâmetros de identificação do modelo."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "architecture": "Zero RAM"
        }
