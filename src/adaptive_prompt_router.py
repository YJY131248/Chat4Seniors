import numpy as np
from typing import List, Optional
from data_utils import load_json

class AdaptivePromptRouter:
    def __init__(self, dialogs: List[dict], template_path: str = ""):
        self.dialogs = dialogs  # init dialog memory
        self.template = load_json(template_path)  # load prompt template from file
    
    def add_dialog(self, info: dict):
        self.dialogs.append(info)
    
    def update_cognitive_ability(self, windows_size: int = 5, decay_lambda: float = 0.5) -> int:
        if not self.dialogs:
            return 0  # init cognitive ability
        
        # read cognitive ability memory
        cognitive_ability_memory = [int(info["congitive_ability"]) for info in self.dialogs if info["role"] == "user"]
        windows_size = min(windows_size, len(cognitive_ability_memory))
        cognitive_ability_memory = cognitive_ability_memory[-windows_size:]

        # compute time weight (exponential decay)
        time_weights = np.exp(-decay_lambda * np.arange(len(cognitive_ability_memory))[::-1])
        # compute weighted average cognitive ability
        weighted_sum = np.sum(np.array(cognitive_ability_memory) * time_weights)
        normalized_weight = np.sum(time_weights)
        next_cognitive_ability = round(weighted_sum / normalized_weight)

        # bonus rule
        ## Avoid misjudging short-term fluctuations
        if cognitive_ability_memory.count(1) >= 2:
            return max(next_cognitive_ability, 1)
        ## Cognitive ability significantly decreased
        if cognitive_ability_memory.count(2) >= 2:
            # maintaining a high cognitive level
            return 2  
        return next_cognitive_ability

    def generate_prompt(self, windows_size: int = 5, decay_lambda: float = 0.5, cognitive_ability: Optional[int] = None) -> str:
        # build context
        context = ""
        for info in self.dialogs:
            context += "{role}: {text}\n".format(role=info["role"], text=info["text"])
        # get cognitive ability
        if cognitive_ability is None:
            cognitive_ability = self.update_cognitive_ability(windows_size=windows_size, decay_lambda=decay_lambda)
        # generate prompt
        if self.dialogs[-1]["role"] == "user":
            prompt = self.template["assistant"][str(cognitive_ability)].replace("[context content]", context)
        else:
            prompt = self.template["user"][str(cognitive_ability)].replace("[context content]", context)
        return prompt

if __name__ == "__main__":
    # # prompt template test
    # prompt_template = load_json("../config/adaptive_prompt.json")
    # print(prompt_template["assistant"]['0'])
    # print()
    # print(prompt_template["assistant"]['1'])
    # print()
    # print(prompt_template["assistant"]['2'])

    # adaptive prompt router test
    dialogs = [
        {
            "role": "assistant",
            "text": "Hello. This is Care Call.  How are you sleeping?",
        },
        {
            "role": "user",
            "text": "It's me, it's always like that.",
            "congitive_ability": "0"
        },
        {
            "role": "assistant",
            "text": "How did your day go?",
        },
        {
            "role": "user",
            "text": "It's been raining since morning, and I'm not feeling good. That's why I felt so sick.",
            "congitive_ability": "1"
        },
        {
            "role": "assistant",
            "text": "You did. In such a case, why don't you relax with a cup of hot tea?",
        },
        {
            "role": "user",
            "text": "Oh! That's a great idea. I just happen to have water in the coffee pot, so I need to drink it.",
            "congitive_ability": "1"
        },
    ]
    rt = AdaptivePromptRouter(dialogs=dialogs, template_path="../config/adaptive_prompt.json")
    print(rt.generate_prompt())
