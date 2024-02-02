# Repeat After Me: Transformers are Better than State Space Models at Copying

## About

This repository gathers the experiments for the paper "Repeat After Me: Transformers are Better than State Space Models at Copying". The experiments divide in two parts: 

- Synthetic experiments: this covers three tasks: standard copy, prefix key veriant of the n-gram lookup task and the suffix key variant. The models we consider are Transformers (with RoPE, NoPE, ALiBi and Hard-ALiBi positional encodings), Mamba and LSTM.

- Experiments with pretrained models: this covers three tasks: copying C4 text, lookup on phone books and question answering on SQuAD_v2.

## Installation

<tt>pip install causal-conv1d>=1.1.0</tt> : an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
<tt>pip install mamba-ssm</tt> : the core Mamba package.
<tt>pip install names</tt> : names package to randomly sample names in the phone-book experiment.

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
- transformers 4.35+
- datasets 2.14+

## Synthetic experiments

<tt>python3 synthetic_tasks/main.py --model  \
                                --train_task $TRAIN \
                                --eval_task  \
                                --num_masked_heads ${NUM_MASKED_HEADS} \
                                --min_train_len $MIN_TRAIN_LEN\
                                --max_train_len $MAX_TRAIN_LEN\
                                --min_eval_len $MIN_EVAL_LEN\
                                --max_eval_len $MAX_EVAL_LEN\
                                --context_len $CONTEXT_LEN\
                                --eval_context_len $EVAL_CONTEXT_LEN\
                                --n_gram $N_GRAM\
                                --length_answer $ANS_LEN\
                                --vocab_size $VOCAB_SIZE\
                                --state_dim $STATE_DIM</tt>
