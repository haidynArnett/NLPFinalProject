# src/experiments/output_task.py
# Implementation of the output task experiment
# Tests structured output parsing and context preservation

import json
from typing import List, Dict, Any

PROMPT = """
You are one node in a sequential pipeline.

Rules:
- Follow the function provided in the task field.
- Put the value in result.
- Output ONLY the JSON object

Return ONLY valid JSON (no extra text) with this exact structure:
{
  "result": integer
}

PRIOR CONTEXT:

"""

class OutputTest:
    def __init__(self, model):
        """
        model   : OllamaClient
        """
        self.model = model
        self.state = 0
        self.MAX_RETRIES = 5
        self.output_format = {
            "type": "object",
            "properties": {
                "result": {"type": "integer"}
            },
            "required": ["result"]
        }

    def _call_model(self, text: str) -> str:
        return self.model.generate_text(text, format=self.output_format)

    def step(self):
        """
        Returns one step.
        """
        retry = 0
        while retry < self.MAX_RETRIES:
            step = {
                "task": "increase the integer in context by 1",
                "context": self.state
            }
            prompt_text = PROMPT + json.dumps(step, ensure_ascii=False)
            raw = self._call_model(prompt_text)
            
            try:
                parsed = json.loads(raw)
                if "result" in parsed:
                    self.state = parsed["result"]
                    break
                else:
                    self.model.remove_last_conversation()
            except Exception as e:
                self.model.remove_last_conversation()
                print(f"Error parsing JSON: {e}")
            
            retry += 1
            
        if retry == self.MAX_RETRIES:
            raise Exception("Max retries reached")
    
    def run(self, steps: int, save_history_every: int = 1, experiment_name: str = "default"):
        for i in range(steps):
            try:
                self.step()
            # Error in step, flush and break
            except Exception as e:
                print(f"Error in step {i}: {e}")
                self.model.flush_conversation_history(experiment_name, True)
                break

            # save history every "save_history_every" steps
            if (i+1) % save_history_every == 0:
                self.model.flush_conversation_history(experiment_name, True)


def run_batch_output_test(
    clients: List[Any],
    num_runs: int = 100,
    iterations_per_run: int = 50,
    save_history_every: int = 50,
    base_experiment_name: str = "output_task"
):
    """
    Run output test multiple times for each client.
    
    Args:
        clients: List of OllamaClient instances
        num_runs: Number of runs per client
        iterations_per_run: Iterations per run
        save_history_every: Flush frequency
        base_experiment_name: Base folder name
    """
    for client in clients:
        model_name = client.model_name
        
        for run_id in range(num_runs):
            experiment_name = f"{base_experiment_name}_{model_name}/run_{run_id:03d}"
            
            try:
                test = OutputTest(client)
                test.run(
                    steps=iterations_per_run,
                    save_history_every=save_history_every,
                    experiment_name=experiment_name
                )
            except Exception:
                pass
            finally:
                client.clear_conversation_history()
