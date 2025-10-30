# src/io/ollama_client.py
# Ollama API client for model inference
# Handles model loading, inference requests, and response parsing
from typing import List, Optional, Sequence, Dict, Any
import ollama
import logging
import json

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def _extract_model_names(models_obj) -> List[str]:
    names: List[str] = []
    if not models_obj:
        return names
    # Handle dict or object response forms
    if isinstance(models_obj, dict) and "models" in models_obj:
        iterable = models_obj["models"]
    elif hasattr(models_obj, "models"):
        iterable = getattr(models_obj, "models")
    else:
        iterable = models_obj
    for m in iterable or []:
        name: Optional[str] = None
        if isinstance(m, dict):
            name = m.get("name") or m.get("model") or m.get("id")
        else:
            for attr in ("name", "model", "id"):
                if hasattr(m, attr):
                    val = getattr(m, attr)
                    if isinstance(val, str) and val:
                        name = val
                        break
        if name:
            names.append(name)
    return names

# Check if model is available locally without pulling
def is_model_available(model_name: str, client: Optional[ollama.Client] = None) -> bool:
    try:
        listed = client.list() if client else ollama.list()
    except Exception as exc:
        logger.error("Failed to list local Ollama models: %s", exc)
        return False
    names = _extract_model_names(listed)
    if model_name in names:
        return True
    # If user passed untagged name, allow match against any tag of the same base
    if ":" not in model_name:
        base = model_name
        for n in names:
            if isinstance(n, str) and n.split(":", 1)[0] == base:
                return True
    return False

def _pull_model(model_name: str, client: Optional[ollama.Client] = None) -> None:
    """Pull a model from the remote registry."""
    logger.info(f"Pulling model '{model_name}' from remote registry...")
    try:
        if client is not None and hasattr(client, "pull") and callable(getattr(client, "pull")):
            client.pull(model=model_name)
        else:
            ollama.pull(model_name)
        logger.info(f"Successfully pulled model '{model_name}'")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 404:
            raise ValueError(f"Model '{model_name}' does not exist on the remote registry") from exc
        logger.error(f"Failed to pull model '{model_name}': {exc}")
        raise

class OllamaClient:
    def __init__(self, model_name: str, host: Optional[str] = None, headers: Optional[dict] = None, log_conversations: bool = True):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the model to use
            host: Optional custom host URL
            headers: Optional custom headers
            log_conversations: Whether to store conversation history (default: True)
        """
        self.model_name = model_name
        self.client = ollama.Client(host=host, headers=headers or {})
        self.log_conversations = log_conversations
        self._conversation_history: List[Dict[str, Any]] = []
        
        if not is_model_available(model_name, self.client):
            logger.info(f"Model '{model_name}' not found locally, attempting to pull...")
            _pull_model(model_name, self.client)

    def generate_text(self, prompt: str) -> str:
        """
        Generate text response from a prompt.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            The generated text response (content only)
        """
        resp = self.client.generate(model=self.model_name, prompt=prompt)
        
        # Log the full conversation if enabled
        if self.log_conversations:
            conversation_entry = {
                "prompt": prompt,
                "response": resp.__dict__,
                "type": "generate"
            }
            self._conversation_history.append(conversation_entry)
        
        # Extract and return only the response content
        if isinstance(resp, dict):
            if "response" in resp and isinstance(resp["response"], str):
                return resp["response"]
            # Fallback if using chat-like structure
            msg = resp.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return content
        
        # Handle object-style response
        if hasattr(resp, "response"):
            return str(resp.response)
        
        # Last resort: convert to string
        return str(resp)
    
    def generate_embeddings(self, text: str) -> List[float]:
        resp = self.client.embed(model=self.model_name, input=text)
        if isinstance(resp, dict):
            if "embedding" in resp:
                return resp["embedding"]
            if "embeddings" in resp and resp["embeddings"]:
                return resp["embeddings"][0]
        raise ValueError("Unexpected embedding response format for single input")
    
    def generate_embeddings_batch(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self.client.embed(model=self.model_name, input=list(texts))
        if isinstance(resp, dict) and "embeddings" in resp:
            return resp["embeddings"]
        if isinstance(resp, dict) and "embedding" in resp:
            return [resp["embedding"]]
        raise ValueError("Unexpected embedding response format for batch input")
    
    # Conversation history accessor methods
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.
        
        Returns:
            List of conversation entries with prompts and full responses
        """
        return self._conversation_history.copy()
    
    def get_last_conversation(self) -> Optional[Dict[str, Any]]:
        """
        Get the last conversation entry.
        
        Returns:
            The most recent conversation entry, or None if history is empty
        """
        return self._conversation_history[-1].copy() if self._conversation_history else None
    
    def clear_conversation_history(self) -> None:
        """Clear all conversation history."""
        self._conversation_history.clear()
    
    def get_conversation_count(self) -> int:
        """
        Get the number of conversations stored.
        
        Returns:
            Number of conversation entries
        """
        return len(self._conversation_history)
    
    def get_last_response_text(self) -> Optional[str]:
        """
        Get the plaintext response from the last conversation.
        
        Returns:
            The response text from the last conversation, or None if history is empty
        """
        if not self._conversation_history:
            return None
        
        last_conv = self._conversation_history[-1]
        response = last_conv.get("response")
        
        if not response:
            return None
        
        # Handle dict response
        if isinstance(response, dict):
            if "response" in response:
                return response["response"]
            # Fallback for chat-like structure
            msg = response.get("message") or {}
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        
        return None