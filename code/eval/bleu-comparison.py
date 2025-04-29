import json
import nltk

def load_json_file(file_path):
  try:
    with open(file_path, 'r') as file:
      return json.load(file)
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
  except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON - {e}")

if __name__ == "__main__":
  input_file_path = "../../input_data/dataset.json"
  input_data = load_json_file(input_file_path)
  input_data_d = {}
  for item in input_data:
      input_data_d[item['id']] = item['annotated_code']

  output_data_claude_path = "../../output/claude/output.json"
  output_data_claude = load_json_file(output_data_claude_path)
  claude_bleu_scores = []
  for output_item in output_data_claude:
    reference_text = input_data_d[output_item['id']].split()
    candidate_text = output_item['llm_annotated_code'].split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_text], candidate_text)
    claude_bleu_scores.append(BLEUscore)
    
  print("Min Claude BLEU Score: " + str(min(claude_bleu_scores)))
  print("Max Claude BLEU Score: " + str(max(claude_bleu_scores)))
  print("Mean Claude BLEU Score: " + str(sum(claude_bleu_scores)/len(claude_bleu_scores)))
  print()

  output_data_gpt4_path = "../../output/gpt-4/output.json"
  output_data_gpt4 = load_json_file(output_data_gpt4_path)
  gpt4_bleu_scores = []
  for output_item in output_data_gpt4:
    reference_text = input_data_d[output_item['id']].split()
    candidate_text = output_item['llm_annotated_code'].split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_text], candidate_text)
    gpt4_bleu_scores.append(BLEUscore)
  
  print("Min GPT4 BLEU Score: " + str(min(gpt4_bleu_scores)))
  print("Max GPT4 BLEU Score: " + str(max(gpt4_bleu_scores)))
  print("Mean GPT4 BLEU Score: " + str(sum(gpt4_bleu_scores)/len(gpt4_bleu_scores)))
  print()