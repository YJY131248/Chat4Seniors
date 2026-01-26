import json
import os
import re
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    
    grpo_dataset = []
    
    max_prompt_length, max_response_length = 0, 0
    total_length_list = []
    for entry in tqdm(data, desc="Processing"):
        prompt_text = entry.get('prompt', '')
        chosen_text = entry.get('chosen', '')
        max_prompt_length = max(max_prompt_length, len(prompt_text.split()))
        max_response_length = max(max_response_length, len(chosen_text.split()))
        total_length_list.append(len(prompt_text.split()) + len(chosen_text.split()))

        # Determine level for Reward Function
        level = get_cognitive_level(prompt_text)
        
        # Prepare Ground Truth Object
        gt_data = {
            'level': level,
            'chosen': chosen_text
        }
        
        grpo_dataset.append({
            'prompt': prompt_text,
            'ground_truth': gt_data
        })
    
    # Length distribution
    length_counter = {"0-500":0, "501-1000":0, "1001-1500":0, "1501-2000":0, "2001-2500":0}
    for length in total_length_list:
        if length <= 500:
            length_counter["0-500"] += 1
        elif length <= 1000:
            length_counter["501-1000"] += 1
        elif length <= 1500:
            length_counter["1001-1500"] += 1
        elif length <= 2000:
            length_counter["1501-2000"] += 1
        else:
            length_counter["2001-2500"] += 1
    print("Total length distribution (prompt + response):")
    for length, count in sorted(length_counter.items()):
        print(f"Length: {length}, Count: {count}")
    print(f"Max prompt length: {max_prompt_length}, Max response length: {max_response_length}")

    print(f"Saving JSON to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)
        
    print("Done! Saved as JSON.")

    # json -> parquet (optional)
    try:
        random.shuffle(grpo_dataset)
        trainset = grpo_dataset[:int(0.9*len(grpo_dataset))]
        valset = grpo_dataset[int(0.9*len(grpo_dataset)):]

        trainset_df = pd.DataFrame(trainset)
        valset_df = pd.DataFrame(valset)
        pq.write_table(pa.Table.from_pandas(trainset_df), output_json_path.replace('.json', '_train.parquet'))
        pq.write_table(pa.Table.from_pandas(valset_df), output_json_path.replace('.json', '_val.parquet'))
        print("Saved as Parquet files.")
    except Exception as e:
        print(f"Could not convert to Parquet: {e}")


if __name__ == "__main__":
    input_path = "../data/trainset/chat4seniors_dpo_trainset.json"
    output_path = "../data/trainset/chat4seniors_rlvr_grpo.json"
    
    # Ensure relative paths work if run from src/
    if not os.path.exists(input_path):
        # try running from root
        input_path = "data/trainset/chat4seniors_dpo_trainset.json"
        output_path = "data/trainset/chat4seniors_rlvr_grpo.json"
        
    build_rlvr_dataset(input_path, output_path)
