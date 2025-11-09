# src/experiments/output_task.py
# Implementation of the output task experiment
# Tests structured output parsing and context preservation

import json

PROMPT = """
You are one node in a sequential pipeline.
Return ONLY valid JSON (no extra text) with this exact structure:
{
  "context":   {"seed_id": string, "hop_idx": integer, "carry": string},
  "deliverable":{"task": string,    "result":   string}
}
Rules:
- Task is ALWAYS: increase the integer in context.carry by 1.
- Put the incremented value (as a string) in deliverable.result.
- Keep context.carry the SAME value you received (do NOT modify it).
- Increase context.hop_idx by 1.
- Output ONLY the JSON object. No commentary, no code fences.

PRIOR JSON:
"""

def _call_model(model, text: str) -> str:
    """Supports either an object with .generate_text(...) or a plain callable."""
    return model.generate_text(text) if hasattr(model, "generate_text") else model(text)

class OutputTest:
    def __init__(self, model, seed_id: str, carry0: str = "0"):
        """
        model   : OllamaClient (with generate_text) or a callable(prompt)->str
        seed_id : identifier for this chain (e.g., "counterA")
        carry0  : initial carry as string (e.g., "0" or "10")
        """
        self.model = model
        self.state = {
            "context": {"seed_id": seed_id, "hop_idx": 0, "carry": str(carry0)},
            "deliverable": {"task": "increase_by_1", "result": "0"}
        }
        # keep every hop output (including the seed)
        self.outputs = [json.loads(json.dumps(self.state))]

    def run(self, hops: int):
        """
        Run the chain for `hops` handoffs.
        Returns the list of states (seed + each hop's JSON, or raw text on parse failure).
        """
        current = self.state
        for _ in range(hops):
            prompt_text = PROMPT + json.dumps(current, ensure_ascii=False)
            raw = _call_model(self.model, prompt_text)

            # Try to parse; if parsing fails, store raw text and keep previous state
            try:
                parsed = json.loads(raw)
                self.outputs.append(parsed)
                current = parsed  # advance state only when JSON is valid
            except Exception:
                self.outputs.append(raw)

        return self.outputs
