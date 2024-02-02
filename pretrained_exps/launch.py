import os

import sys
import subprocess
import itertools
import numpy as np
import random 
import time
import re

##model
MODEL_tab = ["state-spaces/mamba-370m"]#,"EleutherAI/pythia-410m"]

EVAL_TASK_tab = ["phone_book"]
TEXT_ORDER_tab = ["standard"]

EVAL_MIN_LEN_tab = [20]
EVAL_MAX_LEN_tab = [20]



## BATCH SIZE
BS_EVAL_tab = [10]
NUM_BATCHES_tab = [3]



list_param=[
        MODEL_tab,
        EVAL_TASK_tab,
        TEXT_ORDER_tab,
        EVAL_MIN_LEN_tab,
        EVAL_MAX_LEN_tab,
        BS_EVAL_tab,
        NUM_BATCHES_tab,
        ]

list_param = list(itertools.product(*list_param))



for l in list_param:

   MODEL = l[0]
   EVAL_TASK = l[1]
   TEXT_ORDER = l[2]
   EVAL_MIN_LEN = l[3]
   EVAL_MAX_LEN = l[4]
   BS_EVAL = l[5]
   NUM_BATCHES = l[6]

   subprocess.call(['sbatch', 'dell.slurm', 
       str(MODEL),
       str(EVAL_TASK),
       str(TEXT_ORDER), 
       str(EVAL_MIN_LEN), 
       str(EVAL_MAX_LEN),
       str(BS_EVAL),
       str(NUM_BATCHES)
    ])


print('done')
