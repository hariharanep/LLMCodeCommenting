import openai

class Execution:

    def __init__(self, temperature=0.3, top_p=0.2, args=None):
        self.temperature = temperature
        self.top_p = top_p
        self.openai_key = args.openai_key
        self.openai_base = args.openai_base

        openai.api_key = self.openai_key
        openai.api_base = self.openai_base

    def execute(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert software engineer and code annotator. Your task is to add "
                    "clear, concise, technically inline comments to the code in the methods for the provided Python classes, without "
                    "modifying the existing code or docstrings. In your annotations, emphasize identifying and explaining the contextual "
                    "dependencies such as: third-party library functions or classes, class attributes and their "
                    "usage across methods, calls to other class methods within the same class, external/global variables, "
                    "or configurations that the method relies on. If any behavior is ambiguous, make cautious inferences "
                    "but do not fabricate functionality. "
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return response.choices[0]["message"]["content"]