# LLMCodeCommenting

We took our Python class examples from [ClassEval](https://github.com/FudanSELab/ClassEval).
~~~
# Install python dependencies
pip install -r requirements.txt

# Create GPT-4 commented code
cd code
python main.py --openai_key <insert openai_key> --openai_base https://api.openai.com/v1

# Create Claude commented code
cd code 
python main.py --claude_key <insert claude_api_key>

# Calculate the BLEU scores
cd code/eval
python bleu-comparison.py

# Calculate the human evaluation scores
cd code/eval
python human-eval-scores.py
~~~
~~~
├── README.md
├── code
│   ├── BLEU.py
│   ├── claude
│   │   └── execution.py              # Creates Claude generated comments
│   ├── eval
│   │   ├── bleu-comparison.py        # Calculates BLEU scores
│   │   └── human-eval-scores.py      # Calculates human evaluation scores
│   ├── gpt4
│   │   └── execution.py              # Creates GPT-4 generated comments
│   └── main.py
├── input_data
│   └── dataset.json                  # Human commented ClassEval code
├── milestone_assignments
│   ├── ProgressReport.pdf            # Project Progress Report
│   ├── ProjectDemoPresentation.pptx  # Demonstration slides
│   ├── Research on AI LLM Commenting Existing Code.pdf    # Final Paper
│   └── Team Project Proposal.pdf     # Initial project proposal

├── output
│   ├── claude
│   │   ├── human_eval.json           # Human evaluation database for Claude code
│   │   └── output.json               # Claude generated code
│   └── gpt-4
│       ├── human_eval.json           # Human evaluation database for GPT-4 code
│       └── output.json               # GPT-4 generated code
└── requirements.txt                  # Python module versions
~~~
# Results

| LLM                      | GPT-4 | Claude |
| ------------------------ | ----- | ------ |
| Average Human Eval Score | 0.82  | 0.93   |
| Average BLEU Score       | 0.65  | 0.67   |