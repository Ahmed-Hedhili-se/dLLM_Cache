import os
from transformers import AutoConfig

config = AutoConfig.from_pretrained("inclusionAI/LLaDA-MoE-7B-A1B-Instruct", trust_remote_code=True)
print(config)
