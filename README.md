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

These experiments are intended to study a) how well the models learn the copy task in distribution b) the length generalization ability of these models c) their performance in lookup tasks where we give a prefix or suffix n-grams. 

This folder covers three tasks: <tt>copy</tt>, <tt>prefix_ngram</tt>, <tt>suffix_ngram</tt> and using three models: Transformers with different positional encodings (<tt>model = T_rope</tt>, <tt>T_nope</tt>, <tt>T_alibi</tt>,  <tt>T_hard_alibi</tt>), Mamba (<tt>mamba</tt>) and LSTMs (<tt>lstm</tt>). For instance, to run an experiment where we train a Transfomer with RoPE positional encoding on the copy task for strings with length up to 20 and then evalute it on strings of length 20, this is the command to run:

```
python3 synthetic_tasks/main.py --model "T_rope" --train_task "copy" --eval_task  "copy" --min_train_len 5 --max_train_len 20 --min_eval_len 20 --max_eval_len 20
                               
```


## Experiments on pre-trained models

These experiments cover three different tasks: copying natural text strings from the C4 dataset (<tt> eval_task = c4_copy</tt>), lookup on a phone-book (<tt> eval_task = phone_book</tt>) and question answering on squad_v2 (<tt> eval_task = squad</tt>). We consider in particular the following models: 

- Mamba models: <tt> state-spaces/mamba-370m </tt>, <tt> state-spaces/mamba-1.4b </tt>, <tt> state-spaces/mamba-2.8b  </tt>

- Transformers: <tt> EleutherAI/pythia-410m</tt>, <tt> EleutherAI/pythia-1.4b </tt>, <tt> EleutherAI/pythia-2.8b </tt>

For instance, to run an experiment where we evaluate a Mamba-370m on the phone-book dataset with 20 (name,phone-number) entries, we run: 

```
python3 pretrained_exps/main.py --model "state-spaces/mamba-370m" \
                --eval_task "phone_book" \
                --min_eval_len 20\
                --max_eval_len 20\

```



