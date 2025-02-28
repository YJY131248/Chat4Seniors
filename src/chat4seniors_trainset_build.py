import json
from tqdm import tqdm
from inference import get_peft_llm_model_tokenizer, get_llm_response
import warnings
warnings.filterwarnings("ignore")

# load llm & tokenizer
llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(
    "Mistral", 
    "../model/car_model/mistral_lora_model", 
    peft_model_path="../out/car_model/car_llama_lora_model/checkpoint-1285", 
    peft_type="lora",
    use_peft_model=True
)

# dataset filter
def dataset_filter(ipt_json_path: str, opt_json_path: str):
    data = json.load(open(ipt_json_path, "r"))
    save_data = []
    for idx in tqdm(range(len(data))):
        dialog_data = []
        try:
            for info in data[idx]["data"]:
                if info["role"] == 'user':
                    # get user congitive ability
                    cr = get_llm_response(
                        [info["text"]], llm_model, llm_tokenizer, 
                        max_new_tokens=5, 
                        top_p=0.1, 
                        temperature=0.1,
                        repetition_penalty=1.2,
                        do_sample=False
                    )
                    if cr[info["text"]] in ['0', '1', '2']:
                        info["congitive_ability"] = cr
                        dialog_data.append(info)
                    else:   # stop dialog
                        break
                elif info["role"] == 'system':
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
            

if __name__ == "__main__":
    # dataset filter
    dataset_filter(
        ipt_json_path="../data/carecall-corpus/carecall_filtered_10k_trans.json",
        opt_json_path="../data/carecall-corpus/carecall_dialog_dataset_filter.json"
    )
