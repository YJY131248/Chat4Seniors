import csv
import json
import pandas as pd
from tqdm import tqdm
from data_utils import preprocess_text

# Load the car_prompt.json file
PROMPT = json.load(open("../config/car_prompt.json", 'r', encoding='utf-8'))
PROMPT = PROMPT["prompt"]

def alzheimer_dataset_predeal(ipt_path: str, opt_path: str):
    with open(ipt_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
    with open(opt_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['content', 'label'])
        for line in tqdm(lines):
            parts = line.strip().split('\t')
            label = str(parts[1])
            content = preprocess_text(parts[2])
            if parts[0] == 'alzheimer':
                writer.writerow([content, label])
    print("alzheimer dataset has saved in the path:::{}".format(opt_path))

def car_prompt_build(content):
    car_prompt = PROMPT.replace("[User's text input]", content)
    return car_prompt

def sft_trainset_build(ipt_path: str, opt_path: str):
    # Read the dataset
    df = pd.read_csv(ipt_path)
    save_json = []
    for _, row in tqdm(df.iterrows()):
        content = row['content']
        label = row['label']
        car_prompt = car_prompt_build(content)
        save_json.append(
            {
                "instruction": car_prompt,
                "input": "You are a helpful assistant!",
                "output": str(label)
            }
        )
    # Save the dataset
    with open(opt_path, 'w', encoding='utf-8') as json_file:
        json.dump(save_json, json_file, ensure_ascii=True, indent=4)

if __name__ == "__main__":
    # alzheimer dataset predeal
    alzheimer_dataset_predeal(ipt_path="../data/illness/alzheimer.txt", opt_path="../data/illness/alzheimer.csv")
    # sft trainset build
    sft_trainset_build(ipt_path="../data/illness/alzheimer.csv", opt_path="../data/trainset/car_sft_dataset.json")