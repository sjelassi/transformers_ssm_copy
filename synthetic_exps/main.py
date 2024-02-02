import itertools
import os
import wandb
import json
import argparse
from copy import copy
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import re

from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tqdm import tqdm
from collections import Counter 
from pathlib import Path

import string
from model_utils import get_model
from data_utils import get_train_dataset, get_tokenizer, get_eval_dataset
from train_utils import train, save_model
from test_utils import evaluation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()

    ##task
    parser.add_argument('--train_task',choices=["copy","prefix_ngram","suffix_ngram"],
                        required=True, help="tasks to train the model")
    parser.add_argument('--eval_task',choices=["copy","prefix_ngram","suffix_ngram","duplicate_ngram"],
                        required=True, help="tasks to evaluate the model")
    
    parser.add_argument('--vocab_size', default=26, type=int, help="vocabulary size in the strings. maximum is 26.")
    
    parser.add_argument('--n_gram', default=0, type=int, 
            help='''length of the n-gram when training/evaluating on 'prefix_ngram', 'suffix_ngram', 'duplicate_ngram'.
            Set 0 for 'copy' task.''')
    parser.add_argument('--length_answer', default=0, type=int, 
            help="length of the answer to be returned. Set 0 if no constraint on the length of the answer.")

    #model
    parser.add_argument('--model', choices=['T_nope', 'T_rope', 'T_alibi', "T_hard_alibi",  'lstm', 'mamba'],
            required=True, help='''models starting by 'T' are transformers with different positional embeddings. Other choices
            are mamba and lstm.''')
    parser.add_argument('--hidden_size', default=1024, type=int, help="Hidden size of the models")
    parser.add_argument('--layers', default=12, type=int, help="Number of layers in the models.")
    parser.add_argument('--heads', default=16, type=int, help="Number of heads in the transformer models.")
    parser.add_argument('--num_masked_heads', default=8, type=int, help='''Only when model = ''T_hard_alibi''. 
            Number of heads where we apply hard alibi. The remaining heads are set to nope.''')
    parser.add_argument('--state_dim', default=32, type=int, help='''Only when model = ''mamba''. 
            Sets the state dimension of the model.''')

    #optimization
    parser.add_argument('--lr', default=1e-5, type=float, help="choice of learning rate")
    parser.add_argument('--epochs', default=1, type=int, help="number of epochs")
    parser.add_argument('--steps', default=2000, type=int, help="number of steps for each epoch")
    

    parser.add_argument('--train_batch_size', default=8, type=int, help="training batch size")
    parser.add_argument('--eval_batch_size', default=8, type=int, help="evaluation batch size")
    parser.add_argument('--eval_num_batches', default=3, type=int, help='''number of batches to use for evaluation.
            useful to have a mean + std over results.''')
    
    parser.add_argument('--min_train_len', default=5, type=int, help="minimum length of a training example")
    parser.add_argument('--max_train_len', default=20, type=int, help="maximum length of a training example")
    parser.add_argument('--min_eval_len', default=10, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--max_eval_len', default=20, type=int, help="maximum length of an evaluation example")
    

    ##context length
    parser.add_argument('--context_len', default=220, type=int, help="context length during training")
    parser.add_argument('--eval_context_len', default=220, type=int, help="context length at evaluation time")
    
    
    return parser.parse_args()






args = parse_args()

print(args)


## Get train dataset & tokenizer
tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args)
train_dataset = get_train_dataset(args,tokenizer) 




batch = next(iter(train_dataset))

print("-"*100)
print(f"EXAMPLE {batch['input'][0]}")
print("-"*100)
print(batch['input_ids'][-1][batch['mask'][-1]==1], batch['input_ids'][-1], batch['input'][-1])
print("*"*100)

## Get model
model = get_model(args, tokenizer)

print("^"*100)
print(model)
print(f"Number of parameters of the model: {count_parameters(model)}")
print("^"*100)


## train the model
train(args,model,train_dataset,TO_TOKEN)



## save model
save_model(args, model)


## evaluation of the model

print("###EVALUATION")

model.eval()

str_acc_mean_list, str_acc_std_list, char_accuracy_list = evaluation(args, model,tokenizer,TO_TOKEN)


print(args)

print("DONE")



