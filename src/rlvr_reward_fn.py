import re
import math
import torch
from typing import Dict, Any, Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTIVE_PHRASES = [
    "no rush", "take your time", "step by step", "don't worry",
    "that's okay", "perfectly fine", "let's check", "let's try",
    "no hurry", "you're doing great", "good job", "well done",
    "if you don't remember", "that's fine", "no problem"
]

SENSORY_KEYWORDS = [
    "clap", "touch", "look", "listen", "see", "hear",
    "red", "blue", "green", "warm", "cold", "point to", "tap",
    "find", "show me", "bright", "dark", "soft", "loud"
]

CONTEXT_PHRASES = [
    "today", "this morning", "this afternoon", "weather", "sunny", "cloudy",
    "date", "time", "day", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
]

OPEN_ENDED_STARTERS = [
    "how have you", "tell me about", "i'd love to hear", "what do you think",
    "can you share", "would you like to talk", "what was it like"
]

STRUCTURED_QUESTION_STARTERS = [
    "what did you", "did you have", "can you tell me", "do you remember",
    "could you please", "can you confirm"
]

SIMPLE_INSTRUCTIONS = [
    "please", "can you", "let's", "try to", "now", "first", "next"
]


def get_text_metrics(text: str) -> Dict[str, Union[float, int, bool]]:
    if not text:
        return {
            'avg_len': 0.0,
            'question_count': 0,
            'has_support': False,
            'has_sensory': False,
            'has_context': False,
            'has_open_ended': False,
            'has_structured': False,
            'has_simple_instruction': False,
            'num_words': 0,
            'num_sentences': 0
        }

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = len(sentences)
    words = text.split()
    num_words = len(words)

    avg_len = num_words / num_sentences if num_sentences > 0 else 0.0
    question_count = text.count('?')

    text_lower = text.lower()
    has_support = any(p in text_lower for p in SUPPORTIVE_PHRASES)
    has_sensory = any(k in text_lower for k in SENSORY_KEYWORDS)
    has_context = any(c in text_lower for c in CONTEXT_PHRASES)
    has_open_ended = any(o in text_lower for o in OPEN_ENDED_STARTERS)
    has_structured = any(s in text_lower for s in STRUCTURED_QUESTION_STARTERS)
    has_simple_instruction = any(i in text_lower for i in SIMPLE_INSTRUCTIONS)

    return {
        'avg_len': avg_len,
        'question_count': question_count,
        'has_support': has_support,
        'has_sensory': has_sensory,
        'has_context': has_context,
        'has_open_ended': has_open_ended,
        'has_structured': has_structured,
        'has_simple_instruction': has_simple_instruction,
        'num_words': num_words,
        'num_sentences': num_sentences
    }


def calculate_language_complexity_score(avg_len: float, num_words: int, level: int) -> float:
    score = 0.0        
    if level == 1:
        if avg_len > 50:
            score -= 1.0
            
    elif level == 2:
        if avg_len > 30:
            score -= 1.0 
    
    return score


def calculate_interaction_score(metrics: Dict, level: int) -> float:
    score = 0.0
    question_count = metrics['question_count']
    
    if level == 0:
        if metrics['has_open_ended']:
            score += 1
        if 1 <= question_count <= 2:
            score += 1
        elif question_count > 2:
            score -= 1
            
    elif level == 1:
        if metrics['has_structured']:
            score += 1
        if question_count == 1:
            score += 1 
        elif question_count == 2:
            score += 0.5
        elif question_count > 2:
            score -= 1
            
    elif level == 2:
        if metrics['has_sensory']:
            score += 1.0 
        if metrics['has_simple_instruction']:
            score += 1.0
        if question_count <= 1:
            score += 1.0
        elif question_count > 1:
            score -= 2
    
    return score


def calculate_emotional_support_score(metrics: Dict, level: int) -> float:
    score = 0.0
    
    if metrics['has_support']:
        score += 1.0
    else:
        if level == 1:
            score -= 1
        elif level == 2:
            score -= 1.5

    return score


def calculate_context_awareness_score(metrics: Dict, level: int) -> float:
    score = 0.0
    if metrics['has_context']:
        if level == 1:
            score += 1.0
        elif level == 2:
            score += 1.5
    else:
        if level == 1:
            score -= 0.5
        elif level == 2:
            score -= 1.0
    return score


def calculate_task_alignment_score(metrics: Dict, level: int) -> float:
    score = 0.0
    if level == 0:
        if metrics['has_open_ended'] and metrics['num_words'] >= 20:
            score += 1.0
    elif level == 1:
        if metrics['has_structured'] and metrics['has_support']:
            score += 1
    elif level == 2:
        if metrics['has_sensory'] and metrics['has_simple_instruction']:
            score += 1.0

    return score


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Union[Dict, Any],
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    level = 0
    if isinstance(ground_truth, dict):
        level = int(ground_truth.get('level', 0))
    elif hasattr(ground_truth, 'level'):
        level = int(ground_truth.level)

    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1].strip()
    
    metrics = get_text_metrics(solution_str)
    
    lang_score = calculate_language_complexity_score(
        metrics['avg_len'], metrics['num_words'], level
    )
    inter_score = calculate_interaction_score(metrics, level)
    emo_score = calculate_emotional_support_score(metrics, level)
    context_score = calculate_context_awareness_score(metrics, level)
    task_score = calculate_task_alignment_score(metrics, level)
    
    total_score = lang_score + inter_score + emo_score + context_score + task_score
    normalized_score = max(min(total_score, 2), -2)
    return float(normalized_score)


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    from collections import Counter

    with open("/workspace/user_code/Personal/Chat4Seniors/data/trainset/chat4seniors_rlvr_grpo.json", "r", encoding="utf-8") as f:
        trainset = json.load(f)
    
    scores = []
    score_details = []
    
    for item in tqdm(trainset):
        solution_str = item["reward_model"]['ground_truth']["chosen"]
        ground_truth = item["reward_model"]['ground_truth']
        score = compute_score(None, solution_str, ground_truth)
        scores.append(score)
        
        level = ground_truth.get('level', 0)
        metrics = get_text_metrics(solution_str)
        score_details.append({
            'level': level,
            'score': score,
            'avg_len': metrics['avg_len'],
            'question_count': metrics['question_count']
        })
    
    score_counter = Counter([round(s, 1) for s in scores])
    print("\n=== Score Distribution ===")
    for score, count in sorted(score_counter.items()):
        print(f"Score {score}: {count} times")
    
    print("\n=== Score by Level ===")
    for lv in [0, 1, 2]:
        level_scores = [d['score'] for d in score_details if d['level'] == lv]
        if level_scores:
            avg = sum(level_scores) / len(level_scores)
            print(f"Level {lv}: avg={avg:.2f}, min={min(level_scores):.2f}, max={max(level_scores):.2f}, count={len(level_scores)}")
