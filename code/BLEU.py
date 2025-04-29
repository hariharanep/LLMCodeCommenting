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
  file_path = "input_data/dataset.json"
  dataset = load_json_file(file_path)
  gpt4_generated = load_json_file("output/gpt-4/output.json")
  claude_generated = load_json_file("output/claude/output.json")
  gpt4_bleu_average = 0
  claude_bleu_average = 0
  gpt4_count = 0
  claude_count = 0
  if dataset:
    for item in dataset:
      for gpt4 in gpt4_generated:
        if item['id'] == gpt4['id']:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([gpt4['llm_annotated_code']], item['annotated_code'])
          gpt4_bleu_average += BLEUscore
          gpt4_count += 1
          print(f"BLEU score for GPT4{item['id']}: {BLEUscore}")
          break
      for claude in claude_generated:
        if item['id'] == claude['id']:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([claude['llm_annotated_code']], item['annotated_code'])
          claude_bleu_average += BLEUscore
          claude_count += 1
          print(f"BLEU score for Claude{item['id']}: {BLEUscore}")
          break
          print(f"BLEU score for Claude{item['id']}: {BLEUscore}")
          break
    gpt4_bleu_average = gpt4_bleu_average / gpt4_count
    print(f"Average BLEU score for GPT-4: {gpt4_bleu_average}")
    claude_bleu_average = claude_bleu_average / claude_count
    print(f"Average BLEU score for Claude: {claude_bleu_average}")