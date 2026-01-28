import json
import random
from tqdm import tqdm
from typing import List
from adaptive_prompt_router import AdaptivePromptRouter
from llm_api_service import get_ds_llm_response, get_pos_neg_resp_pair
from inference import get_peft_llm_model_tokenizer, get_llm_response
import warnings
warnings.filterwarnings("ignore")

# load llm & tokenizer
llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(
    "Mistral", 
    "../model/car_model", 
    peft_model_path="../out/car_model/car_mistral_lora_model/checkpoint-645", 
    peft_type="lora",
    use_peft_model=True
)

def llm_resp_post_process(llm_resp: str):
    llm_resp = llm_resp.strip()
    llm_resp = llm_resp.replace("\"", "")
    llm_resp = llm_resp.replace("**Assistant:**", "")
    llm_resp = llm_resp.replace("**assistant:**", "")
    llm_resp = llm_resp.replace("Assistant:", "")
    llm_resp = llm_resp.replace("assistant:", "")
    return llm_resp

# dataset filter
def dataset_filter(ipt_json_path: str, opt_json_path: str):
    data = json.load(open(ipt_json_path, "r"))
    save_data = []
    for idx in tqdm(range(len(data))):
        dialog_data = []
        try:
            for info in data[idx]["data"]:
                if info["role"] == 'user':
                    # filter short text
                    if len(info["text"]) <= 20:
                        continue
                    # get user congitive ability
                    cr = get_ds_llm_response(
                        [info["text"]], llm_model, llm_tokenizer, 
                        max_new_tokens=5, 
                        top_p=0.1, 
                        temperature=0.1,
                        repetition_penalty=1.2,
                        do_sample=False
                    )
                    if cr[info["text"]] in ['0', '1', '2']:
                        info["congitive_ability"] = cr[info["text"]]
                        dialog_data.append(info)
                    else:   # stop dialog
                        break
                elif info["role"] == 'system':
                    if len(dialog_data) > 0 and dialog_data[-1]["role"] != "user":
                        continue
                    info["role"] = "assistant"
                    dialog_data.append(info)
        except:
            continue
        # judge whether dialog is valid
        if len(dialog_data) > 2:
            save_data.append({
                "dialog_id": idx,
                "data": dialog_data
            })
    # save data
    json.dump(save_data, open(opt_json_path, "w"), ensure_ascii=False, indent=4)
    print("save data done! save dataset in the path:::{}, total dialog num:::{}".format(opt_json_path, len(save_data)))

def generate_multiturn_dialog(dialogs: List[dict], max_turn: int = 10):
    # remove key: out-of-bounds
    for dialog in dialogs:
        dialog.pop("out-of-bounds")
    # init adaptive prompt router
    apr = AdaptivePromptRouter(dialogs=dialogs, template_path="../config/adaptive_prompt.json")
    # get top-(N-1) dialogs
    while len(apr.dialogs) < max_turn - 1:
        prompt = apr.generate_prompt()
        assistant_r1_resp = get_llm_response(prompt=prompt, endpoint_id="ep-2025hahahahaha")
        role = "assistant" if apr.dialogs[-1]["role"] == "user" else "user"
        resp_text = llm_resp_post_process(assistant_r1_resp["answer_response"])
        assistant_r1_resp = {
            "role": role,
            "text": resp_text
        }
        # compute cognitive ability
        if role == "user":
            cognitive_ability = get_llm_response(
                [assistant_r1_resp["text"]], llm_model, llm_tokenizer, 
                max_new_tokens=5, 
                top_p=0.1, 
                temperature=0.1,
                repetition_penalty=1.2,
                do_sample=False
            )
            cognitive_ability = cognitive_ability[assistant_r1_resp["text"]]
            if cognitive_ability in ['0', '1', '2']:
                assistant_r1_resp.update({"congitive_ability": cognitive_ability})
            else:
                break
        apr.add_dialog(assistant_r1_resp)
    # confirm the last dialog is assistant
    if apr.dialogs[-1]["role"] == "assistant":
        apr.dialogs = apr.dialogs[:-1]
    # build pos & neg dialog pair
    pos_cognitive_ability = apr.update_cognitive_ability()
    neg_cognitive_ability_list = [0, 1, 2]
    neg_cognitive_ability_list.remove(pos_cognitive_ability)
    neg_cognitive_ability = random.choice(neg_cognitive_ability_list)
    pos_prompt = apr.generate_prompt(cognitive_ability=pos_cognitive_ability)
    neg_prompt = apr.generate_prompt(cognitive_ability=neg_cognitive_ability)
    pos_resp, neg_resp = get_pos_neg_resp_pair(pos_prompt, neg_prompt)
    # save multiturn dialogs as ShareGPT format
    multiturn_dialogs = {}
    multiturn_dialogs["conversations"] = []
    multiturn_dialogs["chosen"] = {"from": "gpt", "value": llm_resp_post_process(pos_resp)}
    multiturn_dialogs["rejected"] = {"from": "gpt", "value": llm_resp_post_process(neg_resp)}
    for dialog in apr.dialogs:
        if dialog["role"] == "user":
            multiturn_dialogs["conversations"].append({"from": "human", "value": dialog["text"]})
        else:
            multiturn_dialogs["conversations"].append({"from": "gpt", "value": dialog["text"]})
    return multiturn_dialogs

def dpo_trainset_build(ipt_json_path: str, opt_json_path: str):
    data = json.load(open(ipt_json_path, "r"))
    save_data = []
    for idx in tqdm(range(len(data))):
        try:
            dialogs = data[idx]["data"][:2]
            max_turn = random.choice([3, 5, 7, 9, 11])
            multiturn_dialogs = generate_multiturn_dialog(dialogs, max_turn=max_turn)
            save_data.append(multiturn_dialogs)
        except:
            continue
        # save data
        if (idx + 1) % 100 == 0:
            json.dump(save_data, open(opt_json_path, "w"), ensure_ascii=False, indent=4)
            print("save data done! save dataset in the path:::{}, total dialog num:::{}".format(opt_json_path, len(save_data)))

def dpo_trainset_format_convert_from_sharegpt_to_default(ipt_json_path: str, opt_json_path: str):
    data = json.load(open(ipt_json_path, "r"))
    save_data = []
    for idx in tqdm(range(len(data))):
        conversations = data[idx]["conversations"]
        chosen = data[idx]["chosen"]["value"]
        rejected = data[idx]["rejected"]["value"]
        dialogs = []
        try:
            for conversation in conversations:
                if conversation["from"] == "human":
                    cognitive_ability = get_llm_response(
                        [conversation["value"]], llm_model, llm_tokenizer, 
                        max_new_tokens=5, 
                        top_p=0.1, 
                        temperature=0.1,
                        repetition_penalty=1.2,
                        do_sample=False
                    )
                    cognitive_ability = cognitive_ability[conversation["value"]]
                    dialogs.append({"role": "user", "text": conversation["value"], "congitive_ability": str(cognitive_ability)})
                else:
                    dialogs.append({"role": "assistant", "text": conversation["value"]})
            apr = AdaptivePromptRouter(dialogs=dialogs, template_path="../config/adaptive_prompt.json")
            prompt = apr.generate_prompt()
            save_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        except:
            continue

    # save data
    json.dump(save_data, open(opt_json_path, "w"), ensure_ascii=False, indent=4)
    print("save data done! save dataset in the path:::{}, total dialog num:::{}".format(opt_json_path, len(save_data)))


def build_dpo_hf_eval_dataset():
    with open("../data/trainset/chat4seniors_rlvr_grpo_val.json", "r") as f:
        data = json.load(f)
    save_data = []
    for item in data:
        prompt = "'# Context\n'" + item["prompt"][0]["content"].split('# Context\n')[1]
        save_data.append({
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "reward_model": {
                "ground_truth": {
                    "chosen": item["reward_model"]["ground_truth"]["chosen"],
                    "level": item["reward_model"]["ground_truth"]["level"]
                },
                "style": "rule"
            },
        })
    # save data
    with open("../data/trainset/chat4seniors_dpo_trainset_hf_eval.json", "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    print("save data done! save dataset in the path:::{}, total dialog num:::{}".format("../data/trainset/chat4seniors_dpo_trainset_hf_eval.json", len(save_data)))


if __name__ == "__main__":
    # # dataset filter
    # dataset_filter(
    #     ipt_json_path="../data/carecall-corpus/carecall_filtered_10k_trans.json",
    #     opt_json_path="../data/carecall-corpus/carecall_dialog_dataset_filter.json"
    # )

    # # build dpo trainset(sharegpt_format)
    # dpo_trainset_build(
    #     ipt_json_path="../data/carecall-corpus/carecall_dialog_dataset_filter.json",
    #     opt_json_path="../data/trainset/chat4seniors_dpo_trainset_sharegpt_format.json"
    # )

    # # format convert
    # dpo_trainset_format_convert_from_sharegpt_to_default(
    #     ipt_json_path="../data/trainset/chat4seniors_dpo_trainset_sharegpt_format.json",
    #     opt_json_path="../data/trainset/chat4seniors_dpo_trainset.json"
    # )

    build_dpo_hf_eval_dataset()
