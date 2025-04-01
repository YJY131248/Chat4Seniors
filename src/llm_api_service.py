import multiprocessing
from typing import Tuple
from volcenginesdkarkruntime import Ark

# volcengine client
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    max_retries=5
)

# Ark Volcenging API
def get_ark_volcenging_llm_resp(prompt: str, endpoint_id: str = "", **kwargs)-> dict:
    if endpoint_id == "":
        endpoint_id = "ep-20250206103155-q4fck" # deepseek-r1  

    messages = [{"role": "user", "content": prompt}]
    default_llm_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2000,
    }
    params = {**default_llm_params, **kwargs}

    llm_response_text, think_text, answer_text = "", "", ""
    completion = client.chat.completions.create(
        model=endpoint_id,
        messages = messages,
        **params
    )
    if completion.choices:
        if hasattr(completion.choices[0].message, 'reasoning_content'):
            think_text = completion.choices[0].message.reasoning_content
        answer_text = completion.choices[0].message.content
        if len(think_text) > 0:
            llm_response_text = "<think>\n{}\n</think>\n\n{}".format(think_text.strip(), answer_text.strip())
        else:
            llm_response_text = answer_text.strip()
    
    return {
        "response": llm_response_text,
        "think_response": think_text.strip(),
        "answer_response": answer_text.strip()
    }

# multiprocessing
def get_parallel_llm_resp(prompt: str, result_dict: dict, key: str, endpoint_id: str = "", **kwargs)-> dict:
    result_dict[key] = get_ark_volcenging_llm_resp(prompt=prompt, endpoint_id=endpoint_id, **kwargs)

# DPO pos & neg pair data
def get_pos_neg_resp_pair(pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
    manager = multiprocessing.Manager()
    llm_resp_dict = manager.dict()
    processes = [
        multiprocessing.Process(target=get_parallel_llm_resp, args=(pos_prompt, llm_resp_dict, 'positive', "ep-20250206103155-q4fck")),
        multiprocessing.Process(target=get_parallel_llm_resp, args=(neg_prompt, llm_resp_dict, 'negative', "ep-20250206103155-q4fck"))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    return llm_resp_dict['positive']['answer_response'], llm_resp_dict['negative']['answer_response']
