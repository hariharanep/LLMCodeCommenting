# LLMCodeCommenting

pip install -r requirements.txt

cd code
python main.py --openai_key <insert openai_key> --openai_base https://api.openai.com/v1

cd code 
python main.py --claude_key <insert claude_api_key>

cd code/eval
python bleu-comparison.py