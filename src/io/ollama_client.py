# Ollama API client for model inference

# Derived from https://github.com/ollama/ollama-python
# This class is a wrapper around the Ollama API client for this project and its needs

from typing import List, Optional, Sequence, Dict, Any, TypedDict, Iterator
import ollama
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# TypedDict for conversation entry structure
class ConversationEntry(TypedDict):
    """Structure for a single conversation entry in history."""
    input: str
    response: str
    details: Dict[str, Any]
    timestamp: str

class SavedConversationData(TypedDict):
    """Structure for saved conversation history JSON files."""
    model: str
    experiment_name: str
    saved_at: str
    conversation_count: int
    conversations: List[ConversationEntry]

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
    def __init__(self, model_name: str, host: Optional[str] = None, headers: Optional[dict] = None, 
                 log_conversations: bool = True, output_dir: Optional[str] = None):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the model to use
            host: Optional custom host URL
            headers: Optional custom headers
            log_conversations: Whether to store conversation history (default: True)
            output_dir: Directory path for saving conversation histories (default: "./output")
        """
        self.model_name = model_name
        self.client = ollama.Client(host=host, headers=headers or {})
        self.log_conversations = log_conversations
        self._conversation_history: List[ConversationEntry] = []
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        
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
        
        # Extract response text
        response_text = None
        if isinstance(resp, dict):
            if "response" in resp and isinstance(resp["response"], str):
                response_text = resp["response"]
            else:
                # Fallback if using chat-like structure
                msg = resp.get("message") or {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    response_text = content
        elif hasattr(resp, "response"):
            response_text = str(resp.response)
        else:
            response_text = str(resp)
        
        # Log the full conversation if enabled
        if self.log_conversations:
            # Extract details but exclude context array to save memory
            if hasattr(resp, "__dict__"):
                details = {k: v for k, v in resp.__dict__.items() if k != "context"}
            elif isinstance(resp, dict):
                details = {k: v for k, v in resp.items() if k != "context"}
            else:
                details = resp
            
            conversation_entry: ConversationEntry = {
                "input": prompt,
                "response": response_text,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            self._conversation_history.append(conversation_entry)
        
        return response_text
    
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text input.
        
        Args:
            text: The input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        resp = self.client.embed(model=self.model_name, input=text)
        
        # Handle dict-style response
        if isinstance(resp, dict):
            # Check for 'embeddings' field (from /api/embed endpoint - returns array of arrays)
            if "embeddings" in resp and resp["embeddings"]:
                return resp["embeddings"][0]
            # Check for 'embedding' field (from /api/embeddings endpoint - returns single array)
            if "embedding" in resp:
                return resp["embedding"]
        
        # Handle object-style response (ollama Python library may return objects)
        if hasattr(resp, "embeddings"):
            embeddings = getattr(resp, "embeddings")
            if embeddings and len(embeddings) > 0:
                return list(embeddings[0]) if not isinstance(embeddings[0], list) else embeddings[0]
        
        if hasattr(resp, "embedding"):
            embedding = getattr(resp, "embedding")
            if embedding:
                return list(embedding) if not isinstance(embedding, list) else embedding
        
        # If we get here, log what we received for debugging
        logger.error(f"Unexpected embedding response format")
        logger.error(f"Response type: {type(resp)}")
        logger.error(f"Response value: {resp}")
        if hasattr(resp, "__dict__"):
            logger.error(f"Response attributes: {resp.__dict__}")
        raise ValueError(f"Unexpected embedding response format for single input. Response type: {type(resp)}")
    
    def generate_embeddings_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text inputs.
        
        Args:
            texts: Sequence of input texts to embed
            
        Returns:
            List of embedding vectors (list of lists of floats)
        """
        resp = self.client.embed(model=self.model_name, input=list(texts))
        
        # Handle dict-style response
        if isinstance(resp, dict):
            # Check for 'embeddings' field (standard batch response)
            if "embeddings" in resp and resp["embeddings"]:
                return resp["embeddings"]
            # Handle single embedding wrapped in array
            if "embedding" in resp:
                return [resp["embedding"]]
        
        # Handle object-style response
        if hasattr(resp, "embeddings"):
            embeddings = getattr(resp, "embeddings")
            if embeddings:
                return list(embeddings) if not isinstance(embeddings, list) else embeddings
        
        if hasattr(resp, "embedding"):
            embedding = getattr(resp, "embedding")
            if embedding:
                return [list(embedding) if not isinstance(embedding, list) else embedding]
        
        # If we get here, log what we received for debugging
        logger.error(f"Unexpected embedding response format for batch")
        logger.error(f"Response type: {type(resp)}")
        logger.error(f"Response value: {resp}")
        if hasattr(resp, "__dict__"):
            logger.error(f"Response attributes: {resp.__dict__}")
        raise ValueError(f"Unexpected embedding response format for batch input. Response type: {type(resp)}")
    
    # Conversation history accessor methods
    
    def get_conversation_history(self) -> List[ConversationEntry]:
        """
        Get the full conversation history.
        
        Returns:
            List of conversation entries with prompts and full responses
        """
        return self._conversation_history.copy()
    
    def get_last_conversation(self) -> Optional[ConversationEntry]:
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
        
        return last_conv.get("response")
    
    # Conversation persistence methods
    
    def flush_conversation_history(self, experiment_name: str, clear_after_flush: bool = True) -> Path:
        """
        Save conversation history to a JSON file in the output directory.
        By default, clears the in-memory cache after flushing (useful for large experiments).
        
        Args:
            experiment_name: Name of the experiment (used as folder/file name)
            clear_after_flush: Whether to clear in-memory history after saving (default: True)
            
        Returns:
            Path to the saved JSON file
        """
        # Create experiment directory
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history_{timestamp}.json"
        filepath = experiment_dir / filename
        
        # Prepare data to save
        data: SavedConversationData = {
            "model": self.model_name,
            "experiment_name": experiment_name,
            "saved_at": datetime.now().isoformat(),
            "conversation_count": len(self._conversation_history),
            "conversations": self._conversation_history
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Flushed {len(self._conversation_history)} conversations to {filepath}")
        
        # Clear memory if requested
        if clear_after_flush:
            self.clear_conversation_history()
            logger.info(f"Cleared in-memory history after flush")
        
        return filepath
    
    def load_conversation_history(self, experiment_name: str, filename: Optional[str] = None, load_all: bool = True) -> int:
        """
        Load conversation history from JSON file(s) in the output directory.
        By default, loads ALL files in the experiment directory (useful for large experiments split across flushes).
        
        Args:
            experiment_name: Name of the experiment (folder name)
            filename: Optional specific filename to load. If provided, only loads that file.
            load_all: If True and filename is None, loads all conversation files in chronological order (default: True)
            
        Returns:
            Number of conversations loaded
        """
        experiment_dir = self.output_dir / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        conversation_files = sorted(experiment_dir.glob("conversation_history_*.json"))
        if not conversation_files:
            raise FileNotFoundError(f"No conversation history files found in {experiment_dir}")
        
        # Determine which files to load
        if filename is not None:
            # Load specific file
            filepath = experiment_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            files_to_load = [filepath]
        elif load_all:
            # Load all files in chronological order
            files_to_load = conversation_files
        else:
            # Load only the most recent file
            files_to_load = [conversation_files[-1]]
        
        # Clear current history and load from file(s)
        self._conversation_history = []
        
        for filepath in files_to_load:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
                self._conversation_history.extend(conversations)
        
        if len(files_to_load) == 1:
            logger.info(f"Loaded {len(self._conversation_history)} conversations from {files_to_load[0]}")
        else:
            logger.info(f"Loaded {len(self._conversation_history)} conversations from {len(files_to_load)} files")
        
        return len(self._conversation_history)
    
    def iter_experiment_conversations(self, experiment_name: str) -> Iterator[ConversationEntry]:
        """
        Memory-efficient iterator for processing large experiments.
        Yields one conversation at a time without loading all into memory.
        Use this for embedding generation or analysis on 10,000+ conversations.
        
        Args:
            experiment_name: Name of the experiment (folder name)
            
        Yields:
            Individual ConversationEntry dictionaries with keys: input, response, details, timestamp
            
        Example:
            for conv in client.iter_experiment_conversations("my_experiment"):
                embedding = get_embedding(conv['response'])
                # process one at a time
        """
        experiment_dir = self.output_dir / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        conversation_files = sorted(experiment_dir.glob("conversation_history_*.json"))
        if not conversation_files:
            raise FileNotFoundError(f"No conversation history files found in {experiment_dir}")
        
        for filepath in conversation_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for conversation in data.get("conversations", []):
                    yield conversation