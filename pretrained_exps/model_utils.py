import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import torch

def get_tokenizer():
  return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def get_model(args):
    
    if "mamba" in args.model:
        model = MambaLMHeadModel.from_pretrained(args.model,device="cuda",dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model,device_map="auto")

    return model
