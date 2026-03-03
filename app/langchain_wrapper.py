from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from inference import generate_text, stream_generate_text

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

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """
        Gera tokens em streaming para o LangChain.
        """
        for token in stream_generate_text(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            stop_on_punctuation=True
        ):
            chunk = GenerationChunk(text=token + " ")
            if run_manager:
                run_manager.on_llm_new_token(token + " ")
            yield chunk

    @property
    def _identifying_params(self) -> dict:
        """Parâmetros de identificação do modelo."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "architecture": "Zero RAM"
        }

def init_langchain_wrapper(**kwargs):
    """
    Função auxiliar para inicializar o wrapper ZeroRAMLLM.
    """
    from database_manager import init_db
    from tensor_manager import ensure_v0_weights
    init_db()
    ensure_v0_weights()
    return ZeroRAMLLM(**kwargs)
