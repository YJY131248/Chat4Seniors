import os
import logging
import warnings
import numpy as np
import pandas as pd
import re
import statistics
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoTokenizer
from sklearn.metrics import classification_report
from data_utils import get_alpaca_dataset, get_qa_dataset
from inference import get_peft_llm_model_tokenizer, get_llm_response

# set merge model arguments
@dataclass
class EvalArguments:
    peft_type: str = field(default="lora")
    task_type: str = field(default="classification")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/car_lora_qwen2.5_7b")
    peft_model_path: str = field(default="../out/car_lora_model/checkpoint-3000/")
    dataset_path: str = field(default="../data/trainset/car_sft_dataset.json")
    save_eval_res_path: str = field(default="../out/car_qwen_lora_model/eval_res/eval_res.csv")
    log_path: str = field(default="../log/car_model_eval.log")
    use_peft_model: bool = field(default=False)
    max_new_tokens: int = field(default=1024)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.1)
    temperature: float = field(default=0.1)
    repetition_penalty: float = field(default=1.2)

# Custom compute_metrics function for classification and QA tasks
def compute_metrics(
    preds: List[str],
    labels: List[str],
    levels: Optional[List[int]] = None,
    task_type: str = "classification"
):
    # Classification task
    if task_type == "classification":
        # Convert labels  str to int
        preds = [int(pred) for pred in preds]
        labels = [int(label) for label in labels]
        print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, output_dict=True, digits=4)
        return {
            "classification_report": report
        }

    # QA task
    elif task_type == "qa":
        def compute_cognition_metrics(text):
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
                'context_score': context_score,
            }

        def compute_cognition_mean_metrics(metrics_list, prefix=None):
            scores = {}
            complexity_keys = ['avg_sentence_len', 'question_count', 'support_score', 'context_score']
            for key in complexity_keys:
                mean_val = statistics.mean([m[key] for m in metrics_list])
                scores[f"{prefix}_{key}"] = mean_val
            return scores

        # Calculate complexity metrics for preds and refs
        pred_cognition0_metrics_list = []
        pred_cognition1_metrics_list = []
        pred_cognition2_metrics_list = []
        for pred_text, level in zip(preds, levels):
            pred_cognition_metrics = compute_cognition_metrics(str(pred_text))
            if int(level) == 0:
                pred_cognition0_metrics_list.append(pred_cognition_metrics)
            elif int(level) == 1:
                pred_cognition1_metrics_list.append(pred_cognition_metrics)
            elif int(level) == 2:
                pred_cognition2_metrics_list.append(pred_cognition_metrics)
        pred_cognition_metrics = {}
        pred_cognition0_metrics = compute_cognition_mean_metrics(pred_cognition0_metrics_list, prefix="level0")  
        pred_cognition1_metrics = compute_cognition_mean_metrics(pred_cognition1_metrics_list, prefix="level1")  
        pred_cognition2_metrics = compute_cognition_mean_metrics(pred_cognition2_metrics_list, prefix="level2")
        pred_cognition_metrics.update(pred_cognition0_metrics)
        pred_cognition_metrics.update(pred_cognition1_metrics)
        pred_cognition_metrics.update(pred_cognition2_metrics)

        # Filter out empty strings
        def truncate_to_token_limit(text, tokenizer, max_tokens=500):
            encoded = tokenizer.encode_plus(
                text,
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt"
            )
            return tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

        tokenizer = AutoTokenizer.from_pretrained("../model/base_models/roberta-large")
        preds = [truncate_to_token_limit(p, tokenizer) for p in preds]
        refs = [[truncate_to_token_limit(r, tokenizer)] for r in labels]

        # BLEU Score Calculation
        bleu_score = corpus_bleu([[r[0].split()] for r in refs], [p.split() for p in preds])

        # ROUGE Score Calculation
        rough_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        for r, p in zip(refs, preds):
            scores = rough_scorer.score(r[0], p)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        rouge1_score = np.mean(rouge_scores['rouge1'])
        rouge2_score = np.mean(rouge_scores['rouge2'])
        rougeL_score = np.mean(rouge_scores['rougeL'])

        # METEOR Score Calculation
        meteor_scores = [meteor_score([r[0].split()], p.split()) for r, p in zip(refs, preds)]
        meteor_score_mean = np.mean(meteor_scores)

        # BERTScore Calculation
        P, R, F1 = score(
            preds, [r[0] for r in refs], 
            model_type="../model/base_models/roberta-large", 
            num_layers=17, lang="en", verbose=True, device="cuda:1", batch_size=16
        )
        bert_score_precision = np.mean(P.numpy())
        bert_score_recall = np.mean(R.numpy())
        bert_score_f1 = np.mean(F1.numpy())

        return {
            "bleu": bleu_score,
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougeL": rougeL_score,
            "meteor": meteor_score_mean,
            "bert_score_precision": bert_score_precision,
            "bert_score_recall": bert_score_recall,
            "bert_score_f1": bert_score_f1,
            **pred_cognition_metrics
        }

    else:
        raise ValueError("Task type must be either 'classification' or 'qa'")

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    eval_args = HfArgumentParser(
        (EvalArguments)
    ).parse_args_into_dataclasses()[0]

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=eval_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments: ")
    logger.info("eval_args:")
    logger.info(eval_args.__repr__())

    # load the dataset and tokenizer dataset
    # dataset = get_alpaca_dataset(eval_args.dataset_path, test_size=0.1)  # SFT
    dataset = get_qa_dataset(eval_args.dataset_path)  # qa(dpo & grpo)
    logger.info('dataset build successfully!')

    # get llm resp
    # load model
    llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(
        eval_args.llm_model_name,
        eval_args.llm_model_path,
        eval_args.peft_model_path,
        eval_args.peft_type,
        eval_args.use_peft_model
    )
    logger.info('LLMs {} load successfully! LLM path::: {}'.format(eval_args.llm_model_name, eval_args.llm_model_path))

    # get llm resp
    # query_list = [sample['instruction'] for sample in dataset]   # SFT
    query_list = [sample['prompt'] for sample in dataset]     # DPO & grpo (qa)
    # query_list = query_list[:2]  # for test
    llm_response_mp = get_llm_response(
        query_list=query_list,
        model = llm_model,
        tokenizer = llm_tokenizer,
        max_new_tokens=eval_args.max_new_tokens,
        top_p=eval_args.top_p,
        temperature=eval_args.temperature,
        repetition_penalty=eval_args.repetition_penalty,
        do_sample=eval_args.do_sample
    )
    preds = [llm_response_mp[query] for query in query_list]
    # labels = [sample['output'] for sample in dataset]   # SFT
    labels = [sample['chosen'] for sample in dataset]  # DPO & grpo (qa)
    levels = [sample['level'] for sample in dataset]  # DPO & grpo (qa)
    logger.info('LLMs {} get response successfully!'.format(eval_args.llm_model_name))
    logger.info('pred::: {}\nlabels::: {}'.format(preds, labels))
    
    # save preds and labels
    dir_path = os.path.dirname(eval_args.save_eval_res_path)
    os.makedirs(dir_path, exist_ok=True)
    infer_df = pd.DataFrame({
        "preds": preds,
        "labels": labels,
        "levels": levels
    })
    infer_df.to_csv(eval_args.save_eval_res_path, index=False)
    logger.info('preds and labels save successfully! save path::: {}'.format(eval_args.save_eval_res_path))
    
    # eval
    eval_metrics = compute_metrics(
        preds=preds,
        labels=labels,
        levels=levels,
        task_type=eval_args.task_type,
    )
    print(eval_metrics)
    logger.info('eval metrics: {}'.format(eval_metrics))
    logger.info('eval done!')


if __name__ == "__main__":
    main()
