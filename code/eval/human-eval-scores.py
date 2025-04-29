import json

def load_json_file(file_path):
  try:
    with open(file_path, 'r') as file:
      return json.load(file)
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
  except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON - {e}")


if __name__ == "__main__":
  human_eval_claude_path = "../../output/claude/human_eval.json"
  human_eval_claude = load_json_file(human_eval_claude_path)
  claude_scores = []
  for eval in human_eval_claude:
    claude_scores.append((eval['score_1'] + eval['score_2']) / 2)
    
  print("Claude Human Eval Score: " + str(sum(claude_scores)/len(claude_scores)))
  print()

  human_eval_gpt4_path = "../../output/gpt-4/human_eval.json"
  human_eval_gpt4 = load_json_file(human_eval_gpt4_path)
  gpt4_scores = []
  for eval in human_eval_gpt4:
    gpt4_scores.append((eval['score_1'] + eval['score_2']) / 2)
    
  print("GPT4 Human Eval Score: " + str(sum(gpt4_scores)/len(gpt4_scores)))
  print()
