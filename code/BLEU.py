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
  if dataset:
    for item in dataset:
      BLEUscore = nltk.translate.bleu_score.sentence_bleu([item['code']], item['annotated_code'])
      print(f"BLEU score for {item['id']}: {BLEUscore}")