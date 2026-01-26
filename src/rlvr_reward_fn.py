import re
import math
import torch
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

PPL_THRESHOLD = 20.0  # τ_ppl
PPL_MODEL_NAME = "/devdata/yaojinyu/Chat4Seniors/model/base_models/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTIVE_PHRASES = [
    "no rush", "take your time", "step by step", "don't worry",
    "that's okay", "perfectly fine", "let's check", "let's try",
    "no hurry", "you're doing great", "good job", "well done"
]
SENSORY_KEYWORDS = [
    "clap", "touch", "look", "listen", "see", "hear",
    "red", "blue", "green", "warm", "cold", "point to", "tap"
]

_PPL_MODEL = None
_PPL_TOKENIZER = None


def get_ppl_model():
    global _PPL_MODEL, _PPL_TOKENIZER
    if _PPL_MODEL is None or _PPL_TOKENIZER is None:
        _PPL_TOKENIZER = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
        # _PPL_MODEL = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
        _PPL_MODEL = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
        _PPL_MODEL.eval()
    return _PPL_MODEL, _PPL_TOKENIZER


def get_text_metrics(text: str) -> Dict[str, Union[float, int, bool]]:
    if not text:
        return {
            'avg_len': 0.0,
            'question_count': 0,
            'has_support': False,
            'has_sensory': False,
            'num_words': 0
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

    return {
        'avg_len': avg_len,
        'question_count': question_count,
        'has_support': has_support,
        'has_sensory': has_sensory,
        'num_words': num_words
    }


def calculate_perplexity(text: str) -> Optional[float]:
    if not text or not text.strip():
        return None

    model, tokenizer = get_ppl_model()
    if model is None:
        return None

    try:
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.n_positions
        if encodings.input_ids.size(1) > max_length:
            input_ids = encodings.input_ids[:, :max_length]
        else:
            input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is None:
                return None
            return math.exp(loss.item())
    except Exception as e:
        print(f"Error computing PPL: {e}")
        return None


def calculate_r_val(text: str, num_words: int, threshold: float = PPL_THRESHOLD) -> int:
    ppl = calculate_perplexity(text)
    if ppl is not None:
        return 1 if ppl < threshold else -1
    return 1 if num_words >= 3 else -1


def calculate_r_lens(avg_len: float, level: int) -> int:
    if level == 1:
        return -1 if avg_len >= 15 else 0
    elif level == 2:
        return -1 if avg_len >= 10 else 0
    return 0


def calculate_r_inter(question_count: int, has_sensory: bool, level: int) -> int:
    score = 0
    if level == 1:
        if question_count > 2:
            score -= 1
    elif level == 2:
        if question_count > 1:
            score -= 1
        if has_sensory:
            score += 1
    return score


def calculate_r_emo(has_support: bool, level: int) -> int:
    if has_support:
        return 1
    if level == 1:
        return -1
    elif level == 2:
        return -2
    return 0


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Union[Dict, Any],
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    level = 0
    if isinstance(ground_truth, dict):
        level = int(ground_truth.get('level', 0))
    elif hasattr(ground_truth, 'level'):
        level = int(ground_truth.level)

    metrics = get_text_metrics(solution_str)

    r_val = calculate_r_val(solution_str, metrics['num_words'])
    r_lens = calculate_r_lens(metrics['avg_len'], level)
    r_inter = calculate_r_inter(metrics['question_count'], metrics['has_sensory'], level)
    r_emo = calculate_r_emo(metrics['has_support'], level)

    r_cogn = r_lens + r_inter + r_emo
    return float(r_val + r_cogn)


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    from collections import Counter

    with open("../data/trainset/chat4seniors_rlvr_grpo.json", "r", encoding="utf-8") as f:
        trainset = json.load(f)
    scores = []
    for item in tqdm(trainset):
        solution_str = item['ground_truth']["chosen"]
        ground_truth = item['ground_truth']
        score = compute_score(None, solution_str, ground_truth)
        scores.append(score)
        print(f"Response: {solution_str}\nScore: {score}\n")
    score_counter = Counter(scores)
    print("Score Distribution:")
    for score, count in sorted(score_counter.items()):
        print(f"Score {score}: {count} times")
