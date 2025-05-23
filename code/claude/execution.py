import anthropic

class Execution:

    def __init__(self, temperature=0.3, top_p=0.2, args=None):
        self.temperature = temperature
        self.top_p = top_p
        self.claude_key = args.claude_key

        self.client = anthropic.Anthropic(
            api_key= self.claude_key
        )

    def execute(self, prompt):
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219", 
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=20000,
            system=(
                "You are an expert software engineer and code annotator. Your task is to add "
                    "clear, concise, technically inline comments to the code in the methods for the provided Python classes, without "
                    "modifying the existing code or docstrings. In your annotations, emphasize identifying and explaining the contextual "
                    "dependencies such as: third-party library functions or classes, class attributes and their "
                    "usage across methods, calls to other class methods within the same class, external/global variables, "
                    "or configurations that the method relies on. If any behavior is ambiguous, make cautious inferences "
                    "but do not fabricate functionality. "
                ),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text