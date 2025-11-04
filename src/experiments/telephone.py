# src/experiments/telephone.py
# Implementation of the telephone game experiment
# Tests information retention through sequential agent handoffs

prompt = """
Here are your instructions:
 - Restate the text that follows.
 - Try to keep a similar word count for the restatement that the original text had, but it doesn't have to be exactly the same.
 - Make sure to restate using different words, so that the restatement is not exactly the same as the text.
 - Retain the sematic meaning or the text as much as possible in the restatement, given these constraints.

 Here is the text:

"""

class TelephoneTest:
    def __init__(self, model, text):
        self.model = model
        self.text = text
        self.restatements = [text]

    def run(self, iterations):
        working_text = self.text
        for _ in iterations:
            print(f"Restating: {working_text}")
            working_text = self.model(prompt + working_text)
            self.restatements.append(working_text)
    