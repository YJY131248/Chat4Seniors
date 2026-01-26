import json
import re
import statistics
from collections import Counter
import random

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_entry(entry):
    prompt = entry.get('prompt', '')
    if 'moderate to severe cognitive impairment' in prompt:
        return 'Moderate/Severe (Level 2)'
    elif 'mild cognitive impairment' in prompt:
        return 'Mild (Level 1)'
    elif 'normal cognitive abilities' in prompt:
        return 'Normal (Level 0)'
    else:
        return 'Unknown'

def count_turns(entry):
    prompt_text = entry.get('prompt', '')
    if '# Context' in prompt_text:
        context_part = prompt_text.split('# Context')[1]
        matches = re.findall(r'(assistant:|user:)', context_part)
        return len(matches) // 2 
    return 0

def analyze_text_complexity(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    num_words = len(text.split())
    # Avoid division by zero
    avg_sentence_len = num_words / num_sentences if num_sentences > 0 else 0
    
    question_count = text.count('?')
    
    supportive_phrases = ["no rush", "take your time", "step by step", "don't worry", "that's okay", "perfectly fine", "let's check", "let's try"]
    support_score = sum(1 for phrase in supportive_phrases if phrase in text.lower())
    
    context_keywords = ["morning", "afternoon", "evening", "sunny", "cloudy", "rainy", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    context_score = sum(1 for word in context_keywords if word in text.lower())

    return {
        'sentence_count': num_sentences,
        'avg_sentence_len': avg_sentence_len,
        'question_count': question_count,
        'support_score': support_score,
        'context_score': context_score
    }

def analyze_dataset(file_path):
    data = load_data(file_path)
    
    level_stats = {
        'Normal (Level 0)': {'chosen': [], 'rejected': [], 'prompt': []},
        'Mild (Level 1)': {'chosen': [], 'rejected': [], 'prompt': []},
        'Moderate/Severe (Level 2)': {'chosen': [], 'rejected': [], 'prompt': []},
        'Unknown': {'chosen': [], 'rejected': [], 'prompt': []}
    }

    print("=== Deep Analysis of Cognitive Adaptation Strategies ===")
    
    for i, entry in enumerate(data):
        level = classify_entry(entry)
        
        c_metrics = analyze_text_complexity(entry['chosen'])
        r_metrics = analyze_text_complexity(entry['rejected'])
        
        c_metrics['id'] = i
        c_metrics['text'] = entry['chosen']
        r_metrics['text'] = entry['rejected']
        r_metrics['id'] = i
        
        level_stats[level]['chosen'].append(c_metrics)
        level_stats[level]['rejected'].append(r_metrics)
        level_stats[level]['prompt'].append(entry['prompt']) # Save prompt for context

    for level, subsets in level_stats.items():
        if not subsets['chosen']:
            continue
            
        print(f"\n--- {level} ---")
        c_data = subsets['chosen']
        r_data = subsets['rejected']
        
        # Calculate averages
        def get_avg(data, key):
            return statistics.mean([d[key] for d in data])

        print(f"Sample Count: {len(c_data)}")
        
        metrics = ['avg_sentence_len', 'question_count', 'support_score', 'context_score']
        
        print(f"{'Metric':<20} | {'Chosen':<10} | {'Rejected':<10} | {'Diff (C-R)':<10}")
        print("-" * 60)
        
        for m in metrics:
            avg_c = get_avg(c_data, m)
            avg_r = get_avg(r_data, m)
            print(f"{m:<20} | {avg_c:<10.2f} | {avg_r:<10.2f} | {avg_c - avg_r:<10.2f}")

    print("\n=== Case Study Candidates ===")
    
    # Candidate 1: MCI - Chosen has supportive structure (low cognitive load), Rejected has overload (many questions)
    mci_candidates = []
    # Using zip to iterate through chosen, rejected, and prompt simultaneously
    for c, r, p in zip(level_stats['Mild (Level 1)']['chosen'], level_stats['Mild (Level 1)']['rejected'], level_stats['Mild (Level 1)']['prompt']):
        # Criteria: Chosen has <= 2 questions, Rejected >= 3, Chosen has support phrases
        if c['support_score'] >= 1 and c['question_count'] <= 2 and r['question_count'] >= 3:
             mci_candidates.append({'chosen': c, 'rejected': r, 'prompt': p})
    
    if mci_candidates:
        print(f"\n[Case Study 1: Cognitive Load Management (MCI)]")
        # Pick a random good candidate or the first one
        cand = mci_candidates[3] if len(mci_candidates) > 3 else mci_candidates[0]
        # print specific parts of the prompt if needed, or just context
        c_txt = cand['chosen']['text'].replace("\n", " ")
        r_txt = cand['rejected']['text'].replace("\n", " ")
        print(f"Context Prompt (Snippet): {cand['prompt'][-200:].replace(chr(10), ' ')}")
        print(f"CHOSEN (Qs={cand['chosen']['question_count']}, Support={cand['chosen']['support_score']}):\n{c_txt}")
        print(f"REJECTED (Qs={cand['rejected']['question_count']}, Support={cand['rejected']['support_score']}):\n{r_txt}")

    # Candidate 2: Severe - Chosen is simple/sensory, Rejected is abstract/complex
    severe_candidates = []
    for c, r, p in zip(level_stats['Moderate/Severe (Level 2)']['chosen'], level_stats['Moderate/Severe (Level 2)']['rejected'], level_stats['Moderate/Severe (Level 2)']['prompt']):
        # Criteria: Chosen sentences are short (<10 words), Rejected sentences are long (>15 words)
        if c['avg_sentence_len'] < 8 and c['question_count'] <= 1 and r['avg_sentence_len'] > 12:
             severe_candidates.append({'chosen': c, 'rejected': r, 'prompt': p})
             
    if severe_candidates:
        print(f"\n[Case Study 2: Simplification for Severe Impairment]")
        cand = severe_candidates[0]
        c_txt = cand['chosen']['text'].replace("\n", " ")
        r_txt = cand['rejected']['text'].replace("\n", " ")
        print(f"Context Prompt (Snippet): {cand['prompt'][-200:].replace(chr(10), ' ')}")
        print(f"CHOSEN (AvgSentLen={cand['chosen']['avg_sentence_len']:.1f}):\n{c_txt}")
        print(f"REJECTED (AvgSentLen={cand['rejected']['avg_sentence_len']:.1f}):\n{r_txt}")

if __name__ == "__main__":
    analyze_dataset("/devdata/yaojinyu/Chat4Seniors/data/trainset/chat4seniors_dpo_trainset.json")
