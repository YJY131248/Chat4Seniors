import csv
import json
import pandas as pd
from tqdm import tqdm
from data_utils import preprocess_text
from nlpaug.augmenter.word import SynonymAug, ContextualWordEmbsAug
from nlpaug.augmenter.char import KeyboardAug
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

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
            if label == "3":
                label = "2"
            content = preprocess_text(parts[2])
            if parts[0] == 'alzheimer':
                writer.writerow([content, label])
    print("alzheimer dataset has saved in the path:::{}".format(opt_path))

def augment_text(text, augmenter, num_aug=3):
    try:
        augmented = augmenter.augment(text, n=num_aug)
        return list(set(augmented))
    except:
        return [text]

def alzheimer_dataset_augmentation(ipt_path: str, opt_path: str, target_label: str, augment_multiplier: int=5):
    # load data
    df = pd.read_csv(ipt_path)
    target_df = df[df['label'] == target_label].copy()
    original_count = len(target_df)
    print(f"Data Distribution:\n{df['label'].value_counts()}")
    print(f"Target Label Count: {original_count}")

    # augmentation strategy
    synonym_aug = SynonymAug(aug_src='wordnet', aug_max=2)
    contextual_aug = ContextualWordEmbsAug(
        model_path='../model/roberta-large', action="substitute", device="cuda"
    )
    keyboard_aug = KeyboardAug(aug_char_max=2)

    # augmentation process
    augmented_data = []
    for _, row in tqdm(target_df.iterrows()):
        # strategy 1: synonym replacement
        augmented = augment_text(row['content'], synonym_aug)
        augmented_data.extend([(text, target_label) for text in augmented])
        # strategy 2: contextual word embedding
        augmented = augment_text(row['content'], contextual_aug)
        augmented_data.extend([(text, target_label) for text in augmented])
        # strategy 3: keyboard augmentation
        augmented = augment_text(row['content'], keyboard_aug)
        augmented_data.extend([(text, target_label) for text in augmented])
    
    # save augmented data
    augmented_df = pd.DataFrame(augmented_data, columns=['content', 'label'])
    augmented_df = augmented_df.drop_duplicates(subset=['content'])
    required_samples = original_count * augment_multiplier
    augmented_df = augmented_df.sample(
        n = min(required_samples, len(augmented_df)), 
        replace=False, 
        random_state=42
    )
    augmented_df = pd.concat([df, augmented_df])
    augmented_df.to_csv(opt_path, index=False)

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
    # raw alzheimer dataset predeal
    alzheimer_dataset_predeal(ipt_path="../data/illness/alzheimer.txt", opt_path="../data/illness/alzheimer.csv")
    # raw sft trainset build
    sft_trainset_build(ipt_path="../data/illness/alzheimer.csv", opt_path="../data/trainset/car_sft_dataset_base.json")
    # data augmentation
    alzheimer_dataset_augmentation(
        ipt_path="../data/illness/alzheimer_augmentation.csv",
        opt_path="../data/illness/alzheimer_augmentation.csv",
        target_label=0,
        augment_multiplier=5
    )
    # augmentation sft trainset build
    sft_trainset_build(ipt_path="../data/illness/alzheimer_augmentation.csv", opt_path="../data/trainset/car_sft_dataset_augmentation.json")
