# src/experiments/telephone.py
# Implementation of the telephone game experiment
# Tests information retention through sequential agent handoffs

# prompt = """
# Here are your instructions:
#  - Restate the text that follows.
#  - Try to keep a similar word count for the restatement that the original text had, but it doesn't have to be exactly the same.
#  - Make sure to restate using different words, so that the restatement is not exactly the same as the text.
#  - Keep the meening of the restatement the same, despite using different words.
#  - Do not output any text other than the restatement.
 
#  Here is the text:

# """

# prompt = """
# Here are your instructions:
#  - Restate the text that follows.
#  - Try to keep a similar word count for the restatement that the original text had.
#  - Keep the meening of the restatement the same, despite using different words.
#  - ONLY OUTPUT THE RESTATED TEXT AND NOTHING ELSE
 
#  Here is the text:

# """

prompt = """
Here are your instructions:
 - Restate the text that follows.
 - Keep the meening of the restatement the same, despite using different words.
 - ONLY OUTPUT THE RESTATED TEXT AND NOTHING ELSE
 
 Here is the text:

"""

class TelephoneTest:
    def __init__(self,client, text):
        self.client = client
        self.text = text
        self.restatements = [text]

    def run(self, iterations):
        working_text = self.text
        for _ in range(iterations):
            print(f"Restating: {working_text}")
            working_text = self.client.generate_text(prompt + working_text)
            self.restatements.append(working_text)
    