import torch
import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    HfArgumentParser
)
from peft import PeftModel, PeftModelForCausalLM
from dataclasses import dataclass, field
from finetune import get_llm_model_tokenizer


# Define the inference argument class
@dataclass
class InferenceArguments:
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/Qwen2-7B-Instruct")
    peft_type: str = field(default="lora")
    merge_save_path: str = field(default="../out/car_lora_model/checkpoint-3000/")
    use_merge_model: bool = field(default=False)
    log_path: str = field(default="../out/car_model_inference.log")


def get_peft_llm_model_tokenizer(inference_args):
    # set merge model arguments
    if bool(inference_args.use_merge_model):
        llm_model_name = inference_args.llm_model_name
        if inference_args.peft_type == "lora":
            llm_model_path = inference_args.merge_save_path  
        else:
            llm_model_path = inference_args.llm_model_path  
        peft_model_path = inference_args.merge_save_path

        # load the base LLM model and tokenizer
        if llm_model_name == "Qwen" or llm_model_name == "BaiChuan":
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.half
            )
        elif llm_model_name == "ChatGLM":
            model = AutoModel.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.half
            )
        else:
            logger.error("Invalid model: Supported models are Qwen, ChatGLM, BaiChuan")
            raise ValueError("Invalid model: Supported models are Qwen, ChatGLM, BaiChuan")
        
        # set the model parameters
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True

        # load the PEFT model
        if inference_args.peft_type != "lora":
            model = PeftModelForCausalLM.from_pretrained(model, model_id=peft_model_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    
    else:
        model, tokenizer = get_llm_model_tokenizer(inference_args)

    return model, tokenizer
    

def get_llm_response(query_list: list[str], inference_args):
    # load the LLM model and tokenizer
    llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(inference_args)
    llm_model = llm_model.cuda()
    llm_model.eval()
    logger.info('PEFT LLMs {} load successfully! LLM path::: {}'.format(inference_args.llm_model_name, inference_args.merge_save_path))
    # set the response map
    llm_response_mp = {}
    for query in query_list:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        messages = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([messages], return_tensors="pt").to(llm_model.device)
        generate_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "eos_token_id": llm_model.config.eos_token_id,
        }
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            **generate_kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        llm_response_mp[query] = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return llm_response_mp


def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    inference_args = HfArgumentParser(
        (InferenceArguments)
    ).parse_args_into_dataclasses()[0]

    # 设置logger
    logging.basicConfig(
        level=logging.DEBUG,
        filename=inference_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("命令行参数")
    logger.debug("inference_args:")
    logger.debug(inference_args.__repr__())

    # model inference
    query_list = [
        "What is the best way to prevent Alzheimer's disease?",
    ]
    llm_resp = get_llm_response(query_list=query_list, inference_args=inference_args)
    print(llm_resp)
    logger.info('LLMs:::{}; Query:::{}; Response:::{}'.format(inference_args.llm_model_name, str(query_list), str(llm_resp)))


if __name__ == "__main__":
    main()