import json
import os
import re
from tqdm import tqdm

def get_cognitive_level(prompt_text):
    """
    Extracts cognitive level from the system prompt/text.
    Follows logic from multiturn_dialogues_dataset_analysis.py
    """
    if 'moderate to severe cognitive impairment' in prompt_text:
        return '2'
    elif 'mild cognitive impairment' in prompt_text:
        return '1'
    elif 'normal cognitive abilities' in prompt_text:
        return '0'
    else:
        # Default or Unknown
        return '0'

def build_rlvr_dataset(input_json_path, output_json_path):
    print(f"Loading data from {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Found {len(data)} entries. Processing for GRPO...")
    
    dataset_out = []
    
    for entry in tqdm(data, desc="Processing"):
        prompt_text = entry.get('prompt', '')
        chosen_text = entry.get('chosen', '')
        
        # Determine level for Reward Function
        level = get_cognitive_level(prompt_text)
        
        # Prepare Ground Truth Object
        gt_data = {
            'level': level,
            'chosen': chosen_text
        }
        
        dataset_out.append({
            'prompt': prompt_text,
            'ground_truth': gt_data
        })

    print(f"Saving JSON to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_out, f, ensure_ascii=False, indent=2)
        
    print("Done! Saved as JSON.")


if __name__ == "__main__":
    input_path = "../data/trainset/chat4seniors_dpo_trainset.json"
    output_path = "../data/trainset/chat4seniors_rlvr_grpo.json"
    
    # Ensure relative paths work if run from src/
    if not os.path.exists(input_path):
        # try running from root
        input_path = "data/trainset/chat4seniors_dpo_trainset.json"
        output_path = "data/trainset/chat4seniors_rlvr_grpo.json"
        
    build_rlvr_dataset(input_path, output_path)
