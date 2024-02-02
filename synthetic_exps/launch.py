import os

import sys
import subprocess
import itertools
import numpy as np
import random 
import time
import re

##model
MODEL_tab = ["lstm"]
HIDDEN_tab = [1024]
LAYER_tab = [4]
HEAD_tab = [0]


## CLIPPED NOPE specific
NUM_MASKED_HEADS_tab=[0]


## MAMBA STATE specific
STATE_DIM_tab = [0]



##TRAIN & EVAL
TRAIN_tab = ["copy"]
EVAL_tab= ["copy"]

N_GRAM_tab = [0]
ANS_LEN_tab = [0]
VOCAB_SIZE_tab = [26]

##EPOCHS & STSEPS
EPOCH_tab = [1]

STEP_tab = [200]

## LENGTHS
MIN_TRAIN_LEN_tab = [10]
MAX_TRAIN_LEN_tab = [21]
EVAL_MIN_LEN_tab = [10]
EVAL_MAX_LEN_tab = [21]


## CONTEXT LEN
CONTEXT_LEN_tab = [120]
EVAL_CONTEXT_LEN_tab = [120]



## BATCH SIZE
BS_TRAIN_tab = [32]
BS_EVAL_tab = [32]#[64]#64
NUM_BATCHES_tab = [3]#20]#20]




##LR
LR_tab = [5e-5]



list_param= [
        MODEL_tab,
        HIDDEN_tab,
        LAYER_tab,
        HEAD_tab,
        NUM_MASKED_HEADS_tab,
        STATE_DIM_tab,
        TRAIN_tab,
        EVAL_tab,
        N_GRAM_tab,
        ANS_LEN_tab,
        VOCAB_SIZE_tab,
        EPOCH_tab,
        STEP_tab,
        MIN_TRAIN_LEN_tab,
        MAX_TRAIN_LEN_tab,
        EVAL_MIN_LEN_tab,
        EVAL_MAX_LEN_tab,
        CONTEXT_LEN_tab,
        EVAL_CONTEXT_LEN_tab,
        BS_TRAIN_tab,
        BS_EVAL_tab,
        NUM_BATCHES_tab,
        LR_tab,
        ]


list_param = list(itertools.product(*list_param))



for l in list_param:

   MODEL = l[0]
   HIDDEN = l[1]
   LAYER = l[2]
   HEADS = l[3]
   NUM_MASKED_HEADS = l[4]
   STATE_DIM = l[5]
   TRAIN = l[6]
   EVAL = l[7]
   N_GRAM = l[8]
   ANS_LEN = l[9]
   VOCAB_SIZE = l[10]
   EPOCHS = l[11]
   STEPS = l[12]
   MIN_TRAIN_LEN = l[13]
   MAX_TRAIN_LEN = l[14]
   EVAL_MIN_LEN = l[15]
   EVAL_MAX_LEN = l[16]
   CONTEXT_LEN = l[17]
   EVAL_CONTEXT_LEN = l[18]
   BS_TRAIN = l[19]
   BS_EVAL = l[20]
   NUM_BATCHES = l[21]
   LR = l[22]
   
   
   
   subprocess.call(['sbatch', 'dell.slurm',
       str(MODEL),
       str(HIDDEN),
       str(LAYER),
       str(HEADS),
       str(NUM_MASKED_HEADS),
       str(STATE_DIM),
       str(TRAIN),
       str(EVAL),
       str(N_GRAM),
       str(ANS_LEN),
       str(VOCAB_SIZE),
       str(EPOCHS),
       str(STEPS),
       str(MIN_TRAIN_LEN),
       str(MAX_TRAIN_LEN),
       str(EVAL_MIN_LEN),
       str(EVAL_MAX_LEN),
       str(CONTEXT_LEN),
       str(EVAL_CONTEXT_LEN),
       str(BS_TRAIN),
       str(BS_EVAL),
       str(NUM_BATCHES),
       str(LR),
    ])



       
       


print('done')
