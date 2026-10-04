"""
Prompt Compressor for shrinking context size.
"""

from abc import ABC, abstractmethod
from config.logging_config import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class BasePromptCompressor(ABC):
    @abstractmethod
    def compress_context(self, query: str, context_str: str) -> str:
        """
        Compresses the given context string based on the query.
        """
        pass

class LLMLingua2PromptCompressor(BasePromptCompressor):
    def __init__(self):
        try:
            from llmlingua import PromptCompressor as LLMLinguaCompressor
            # Initialize LLMLingua.
            # using a smaller model by default to save memory / time if not specified.
            self.llm_lingua = LLMLinguaCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", 
                use_llmlingua2=True,
                device_map="cpu"
            )
            logger.info("Initialized LLMLingua for prompt compression")
        except ImportError:
            logger.error("LLMLingua not installed. Please install with `uv add llmlingua`.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua: {e}")
            raise
    
    def compress_context(self, query: str, context_str: str) -> str:
        logger.info(f"Compressing context with LLMLingua. Original length: {len(context_str)}")
        try:
            # Target token count based on settings
            target_token = getattr(settings, "LLMLINGUA_TARGET_TOKEN", 200)
            results = self.llm_lingua.compress_prompt_llmlingua2(
                context=[context_str],
                target_token=target_token
            )
            compressed_prompt = results.get("compressed_prompt", context_str)
            origin_tokens = results.get("origin_tokens", "unknown")
            compressed_tokens = results.get("compressed_tokens", "unknown")
            logger.info(f"Context compressed. Tokens before: {origin_tokens}, Tokens after: {compressed_tokens}")
            return compressed_prompt
        except Exception as e:
            logger.error(f"Error during LLMLingua prompt compression: {e}")
            return context_str

class TruncationPromptCompressor(BasePromptCompressor):
    """
    A simple baseline compressor that just truncates context to a maximum character length.
    """
    def __init__(self):
        self.max_chars = getattr(settings, "PROMPT_COMPRESSION_MAX_CHARS", 4000)
        logger.info(f"Initialized Truncation for prompt compression (max_chars={self.max_chars})")

    def compress_context(self, query: str, context_str: str) -> str:
        logger.info(f"Compressing context with Truncation. Original length: {len(context_str)}")
        if len(context_str) > self.max_chars:
            compressed = context_str[:self.max_chars] + "... [TRUNCATED]"
            logger.info(f"Context compressed. New length: {len(compressed)}")
            return compressed
        return context_str


class NoOpPromptCompressor(BasePromptCompressor):
    """
    Fallback compressor that does not compress the prompt.
    """
    def compress_context(self, query: str, context_str: str) -> str:
        return context_str


def get_prompt_compressor() -> BasePromptCompressor:
    """
    Factory method to return the configured prompt compressor.
    """
    method = getattr(settings, "PROMPT_COMPRESSION_METHOD", "").lower()
    
    if method == "llmlingua2":
        return LLMLingua2PromptCompressor()
    elif method == "truncation":
        return TruncationPromptCompressor()
    else:
        logger.warning(f"Unknown or unspecified prompt compression method: '{method}'. Defaulting to No-Op.")
        return NoOpPromptCompressor()
