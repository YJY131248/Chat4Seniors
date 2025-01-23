import logging
import warnings
import numpy as np
import evaluate
from typing import Dict, Union
from dataclasses import dataclass, field
from transformers import (
    AutoModel, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    Trainer, 
    HfArgumentParser, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.metrics import classification_report
from data_utils import get_tokenizer_dataset
from finetune import get_llm_model_tokenizer
from data_utils import get_alpaca_dataset, get_tokenizer_dataset


# set merge model arguments
@dataclass
class EvalArguments:
    peft_type: str = field(default="lora")
    task_type: str = field(default="classification")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/car_lora_qwen2.5_7b")
    dataset_path: str = field(default="../data/trainset/car_sft_dataset.json")
    log_path: str = field(default="../log/car_model_eval.log")
    max_length: int = field(default=1024)


# Custom compute_metrics function for classification and QA tasks
def compute_metrics(tokenizer, task_type: str):
    def fn(eval_pred):
        predictions, labels = eval_pred
        # Classification task
        if task_type == "classification":
            preds = np.argmax(predictions, axis=-1)
            report = classification_report(labels, preds, output_dict=True, digits=4)
            return {
                "classification_report": report
            }

        # QA task
        elif task_type == "qa":
            bleu = evaluate.load('bleu')
            rouge = evaluate.load('rouge')
            meteor = evaluate.load('meteor')
            bertscore = evaluate.load('bertscore')

            # Decode predictions and labels
            preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            refs = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            # Filter out empty strings
            preds = [p if p else " " for p in preds]
            refs = [[r if r else " "] for r in refs]

            # Compute metrics
            bleu_score = bleu.compute(predictions=preds, references=refs)
            rouge_scores = rouge.compute(predictions=preds, references=refs)
            meteor_score = meteor.compute(predictions=preds, references=refs)
            bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")

            return {
                "bleu": bleu_score["bleu"],
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "meteor": meteor_score["meteor"],
                "bert_score_f1": np.mean(bert_score["f1"]),
            }

        else:
            raise ValueError("Task type must be either 'classification' or 'qa'")
    return fn


# Evaluate function using Trainer
def evaluate_model(
    model: Union[AutoModelForCausalLM, AutoModel],
    tokenizer: AutoTokenizer,
    eval_dataset,
    train_args: TrainingArguments,
    task_type: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate model performance using Trainer.
    Args:
        model: Fine-tuned LLM model
        tokenizer: Associated tokenizer
        eval_dataset: Evaluation dataset (Hugging Face Dataset)
        task_type: 'classification' or 'qa'
        device: Computing device
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True, 
        max_length=tokenizer.model_max_length
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(tokenizer, task_type),
        data_collator=data_collator
    )

    # Run evaluation
    eval_results = trainer.evaluate()
    return eval_results


def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    eval_args, train_args = HfArgumentParser(
        (EvalArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        filename=eval_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("Arguments: ")
    logger.debug("eval_args:")
    logger.debug(eval_args.__repr__())

    # load the base LLM model and tokenizer
    llm_model, llm_tokenizer = get_llm_model_tokenizer(eval_args.llm_model_name, eval_args.llm_model_path, eval_args.peft_type)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(eval_args.llm_model_name, eval_args.llm_model_path))

    # load the dataset and tokenizer dataset
    dataset = get_alpaca_dataset(eval_args.dataset_path, test_size=0.1)
    logger.info('dataset build successfully!')
    tokenizer_dataset = get_tokenizer_dataset(dataset, llm_tokenizer, max_length=eval_args.max_length)
    logger.info('tokenizer dataset build successfully!')

    # evaluate the model
    logger.info('Evaluate start!')
    evaluation_results = evaluate_model(
        model=llm_model,
        tokenizer=llm_tokenizer,
        train_args=train_args,
        eval_dataset=tokenizer_dataset["test"],
        task_type=eval_args.task_type
    )
    logger.info(f'Evaluation results: {evaluation_results}')


if __name__ == "__main__":
    main()