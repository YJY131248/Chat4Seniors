import torch
import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    HfArgumentParser
)
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Union
from dataclasses import dataclass, field
from data_utils import get_dpo_dataset


# Define the fine-tuning argument class
@dataclass
class DPOArguments:
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/Qwen2-7B-Instruct")
    dataset_path: str = field(default="../data/alpaca_gpt4_data_zh.json")
    log_path: str = field(default="../log/lora_output.log")
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)

def apply_lora(model, dpo_args):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if dpo_args.llm_model_name == "BaiChuan":
        target_modules = ["W_pack", "W_unpack", "W_proj", "W_o", "W_gate", "W_up", "W_down"]
    lora_config = LoraConfig(
        r=dpo_args.lora_rank,
        lora_alpha=dpo_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=dpo_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# Load LLM model and tokenizer
def get_llm_model_tokenizer(llm_model_name: str, llm_model_path: str, dpo_args: DPOArguments):
    """
    Load base LLM model and tokenizer
    Args:
        llm_model_name: Name of the LLM model
        llm_model_path: Path to the LLM model
    Returns:
        Tuple of (model, tokenizer)
    """

    try:
        if llm_model_name in ["Qwen", "Mistral", "Yi"]:
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif llm_model_name == "Llama":
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path, 
                legacy=True,
                use_fast=False,
                padding_side="right"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif llm_model_name == "BaiChuan":
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        else:
            logger.error(f"Invalid model: Supported models are Qwen, BaiChuan, Mistral, Llama, Yi")
            raise ValueError(f"Invalid model: Supported models are Qwen, BaiChuan, Mistral, Llama, Yi")
        
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True
        model = apply_lora(model, dpo_args)
        return model, tokenizer
    
    except Exception as e:
        raise "Failed to load model: {}".format(e)


# Training function
def dpo_train(
    model: Union[AutoModelForCausalLM, AutoModel],
    tokenizer: AutoTokenizer,
    dataset: Dict,
    train_args: DPOConfig,
    dpo_args: DPOArguments
):
    """
    Direct Preference Optimization (DPO) training
    Args:
        model: Base LLM model
        tokenizer: Tokenizer for text processing
        dataset: Training and evaluation datasets
        train_args: Training arguments
        dpo_args: DPO arguments
    """
    trainer = DPOTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )
    trainer.train()


def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    dpo_args, training_args = HfArgumentParser(
        (DPOArguments, DPOConfig)
    ).parse_args_into_dataclasses()

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=dpo_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments: ")
    logger.info("finetune_args:")
    logger.info(dpo_args.__repr__())
    logger.info("training_args:")
    logger.info(training_args.__repr__())

    # load the base LLM model and tokenizer
    llm_model, llm_tokenizer = get_llm_model_tokenizer(dpo_args.llm_model_name, dpo_args.llm_model_path, dpo_args)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(dpo_args.llm_model_name, dpo_args.llm_model_path))

    # load the dataset and tokenizer dataset
    dataset = get_dpo_dataset(dpo_args.dataset_path, test_size=0.1)
    logger.info('dataset build successfully!')

    # start training
    logger.info('Train start!')
    dpo_train(
        model=llm_model, 
        tokenizer=llm_tokenizer, 
        dataset=dataset, 
        train_args=training_args,
        dpo_args=dpo_args
    )
    logger.info('Train end! LoRA model saves in the path:::{}'.format(training_args.output_dir))


if __name__ == "__main__":
    main()