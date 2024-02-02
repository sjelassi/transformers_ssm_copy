# Repeat After Me: Transformers are Better than State Space Models at Copying

## About

This repository gathers the experiments for the paper "Repeat After Me: Transformers are Better than State Space Models at Copying". The experiments divide in two parts: 

- Synthetic experiments: this covers three tasks: standard copy, prefix key veriant of the n-gram lookup task and the suffix key variant. The models we consider are Transformers (with RoPE, NoPE, ALiBi and Hard-ALiBi positional encodings), Mamba and LSTM.

- Experiments with pretrained models: this covers three tasks: copying C4 text, lookup on phone books and question answering on SQuAD_v2.

## Installation

<tt>pip install causal-conv1d>=1.1.0</tt>
