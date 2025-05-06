# LLMCodeCommenting
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

