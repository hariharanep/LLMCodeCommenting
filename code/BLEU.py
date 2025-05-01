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
  dataset = load_json_file("input_data/dataset.json")
  gpt4_generated = load_json_file("output/gpt-4/output.json")
  claude_generated = load_json_file("output/claude/output.json")
  gpt4_bleu_scores = []
  claude_bleu_scores = []
  if dataset and gpt4_generated and claude_generated:
    for item in dataset:
      found_gpt4 = False
      found_claude = False
      target_code = item['annotated_code']
      print(f"Item ID: {item['id']}")
      gpt4_code = ""
      claude_code = ""
      for gpt4_item in gpt4_generated:
        if gpt4_item['id'] == item['id']:
          gpt4_code = gpt4_item['llm_annotated_code']
          found_gpt4 = True
          break
      for claude_item in claude_generated:
        if claude_item['id'] == item['id']:
          claude_code = claude_item['llm_annotated_code']
          found_claude = True
          break
      if not found_gpt4 or not found_claude:
        print(f"Warning: Missing generated code for item ID {item['id']}")
        continue
      BLEUscore = nltk.translate.bleu_score.sentence_bleu([target_code], gpt4_code)
      gpt4_bleu_scores.append(BLEUscore)
      print(f"BLEU score for GPT4 {item['id']}: {BLEUscore}")
      BLEUscore = nltk.translate.bleu_score.sentence_bleu([target_code], claude_code)
      claude_bleu_scores.append(BLEUscore)
      print(f"BLEU score for Claude {item['id']}: {BLEUscore}")
    gpt4_bleu_scores = np.array(gpt4_bleu_scores)
    claude_bleu_scores = np.array(claude_bleu_scores)
    print(f"Minimum BLEU score for GPT-4: {np.min(gpt4_bleu_scores)}")
    print(f"Maximum BLEU score for GPT-4: {np.max(gpt4_bleu_scores)}")
    print(f"Average BLEU score for GPT-4: {np.mean(gpt4_bleu_scores)}")
    print(f"Minimum BLEU score for Claude: {np.min(claude_bleu_scores)}")
    print(f"Maximum BLEU score for Claude: {np.max(claude_bleu_scores)}")
    print(f"Average BLEU score for Claude: {np.mean(claude_bleu_scores)}")