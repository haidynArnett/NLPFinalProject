# src/experiments/telephone.py
# Implementation of the telephone game experiment
# Tests information retention through sequential agent handoffs

from typing import List, Any

PROMPT = """
Here are your instructions:
 - Restate the text that follows.
 - Keep the meaning of the restatement the same, despite using different words.
 - ONLY OUTPUT THE RESTATED TEXT AND NOTHING ELSE
 
 Here is the text:

"""

DEFAULT_INITIAL_TEXT = "The quick brown fox jumps over the lazy dog."

class TelephoneTest:
    def __init__(self, text_client, embedding_client, initial_text):
        """
        Args:
            text_client: OllamaClient for generating restatements
            embedding_client: OllamaClient for generating embeddings
            initial_text: Starting text for the telephone game
        """
        self.text_client = text_client
        self.embedding_client = embedding_client
        self.initial_text = initial_text
        self.current_text = initial_text
        
        # Save initial text with its embedding as first entry
        initial_embedding = self.embedding_client.generate_embeddings(initial_text)
        self.text_client._conversation_history.append({
            "input": "INITIAL_TEXT",
            "response": initial_text,
            "embedding": initial_embedding,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "details": {}
        })
    
    def step(self):
        """Generate one restatement and compute its embedding."""
        prompt_text = PROMPT + self.current_text
        restatement = self.text_client.generate_text(prompt_text)
        
        # Compute and attach embedding
        embedding = self.embedding_client.generate_embeddings(restatement)
        self.text_client.add_embedding_to_last_conversation(embedding)
        
        self.current_text = restatement
    
    def run(self, steps: int, save_history_every: int = 1, experiment_name: str = "default"):
        """Run telephone test for N iterations."""
        for i in range(steps):
            try:
                self.step()
            except Exception as e:
                print(f"Error in step {i}: {e}")
                self.text_client.flush_conversation_history(experiment_name, True)
                break
            
            if (i+1) % save_history_every == 0:
                self.text_client.flush_conversation_history(experiment_name, True)


def run_batch_telephone_test(
    text_clients: List[Any],
    embedding_client: Any,
    initial_text: str = DEFAULT_INITIAL_TEXT,
    num_runs: int = 100,
    iterations_per_run: int = 50,
    save_history_every: int = 50,
    base_experiment_name: str = "telephone"
):
    """
    Run telephone test multiple times for each client.
    
    Args:
        text_clients: List of OllamaClient instances for text generation
        embedding_client: Single OllamaClient for embeddings
        initial_text: Starting text (defaults to DEFAULT_INITIAL_TEXT)
        num_runs: Number of runs per client
        iterations_per_run: Iterations per run
        save_history_every: Flush frequency
        base_experiment_name: Base folder name
    """
    for client in text_clients:
        model_name = client.model_name
        
        for run_id in range(num_runs):
            experiment_name = f"{base_experiment_name}_{model_name}/run_{run_id:03d}"
            
            try:
                test = TelephoneTest(client, embedding_client, initial_text)
                test.run(
                    steps=iterations_per_run,
                    save_history_every=save_history_every,
                    experiment_name=experiment_name
                )
            except Exception:
                pass
            finally:
                client.clear_conversation_history()
    