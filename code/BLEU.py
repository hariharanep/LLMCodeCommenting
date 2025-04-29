import json
import nltk
import numpy as np

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
  gpt4_bleu_scores = []
  claude_bleu_scores = []
  if dataset:
    for item in dataset:
      for gpt4 in gpt4_generated:
        if item['id'] == gpt4['id']:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([gpt4['llm_annotated_code']], item['annotated_code'])
          gpt4_bleu_scores.append(BLEUscore)
          print(f"BLEU score for GPT4{item['id']}: {BLEUscore}")
          break
      for claude in claude_generated:
        if item['id'] == claude['id']:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([claude['llm_annotated_code']], item['annotated_code'])
          claude_bleu_scores.append(BLEUscore)
          print(f"BLEU score for Claude{item['id']}: {BLEUscore}")
          break
          print(f"BLEU score for Claude{item['id']}: {BLEUscore}")
          break
    gpt4_bleu_scores = np.array(gpt4_bleu_scores)
    claude_bleu_scores = np.array(claude_bleu_scores)
    print(f"Minimum BLEU score for GPT-4: {np.min(gpt4_bleu_scores)}")
    print(f"Maximum BLEU score for GPT-4: {np.max(gpt4_bleu_scores)}")
    print(f"Average BLEU score for GPT-4: {np.mean(gpt4_bleu_scores)}")
    print(f"Minimum BLEU score for Claude: {np.min(claude_bleu_scores)}")
    print(f"Maximum BLEU score for Claude: {np.max(claude_bleu_scores)}")
    print(f"Average BLEU score for Claude: {np.mean(claude_bleu_scores)}")