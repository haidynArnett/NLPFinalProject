# src/io/ollama_client.py
# Ollama API client for model inference
# Handles model loading, inference requests, and response parsing
import ollama


# Check if model is available
def is_model_available(model_name: str) -> bool:
    try:
        ollama.Client().models.pull(model_name)
        return True
    except Exception as e:
        return False

# Pull model
def pull_model(model_name: str) -> None:
    ollama.Client().models.pull(model_name)

class OllamaClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client()

    def generate_text(self, prompt: str) -> str:
        return self.client.generate(model=self.model_name, prompt=prompt)
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        return self.client.embed(model=self.model_name, input=text)
    
    def generate_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        return self.client.embed(model=self.model_name, input=texts)