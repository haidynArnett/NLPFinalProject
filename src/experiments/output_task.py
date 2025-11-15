# src/experiments/output_task.py
# Implementation of the output task experiment
# Tests structured output parsing and context preservation

import json

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

    def _call_model(self, text: str) -> str:
        return self.model.generate_text(text)

    def step(self):
        """
        Returns one step.
        """
        retry = 0
        while retry < self.MAX_RETRIES:
            # create step
            step = {
                "task": "increase the integer in context by 1",
                "context": self.state
            }
            prompt_text = PROMPT + json.dumps(step, ensure_ascii=False)
            raw = self._call_model(prompt_text)
            # check if can be parsed as JSON
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                retry += 1
                continue

            # check if it has a result
            if "result" in parsed:
                self.state = parsed["result"]
                break
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
