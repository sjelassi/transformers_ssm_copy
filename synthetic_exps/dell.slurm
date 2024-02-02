#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner



# _requeue



# _requeue

export MASTER_IP=$(hostname -I | cut -d' ' -f1)

export MASTER_PORT=$(expr 10000 + $(( $RANDOM % 1000 + 1 )) + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * SLURM_GPUS_ON_NODE))


MASTER_IP=$(hostname -I | cut -d' ' -f1)

MODEL=$1
HIDDEN=$2
LAYER=$3
HEADS=$4
NUM_MASKED_HEADS=$5
STATE_DIM=$6
TRAIN=$7
EVAL=$8
N_GRAM=$9
ANS_LEN=${10}
VOCAB_SIZE=${11}
EPOCHS=${12}
STEPS=${13}
MIN_TRAIN_LEN=${14}
MAX_TRAIN_LEN=${15}
MIN_EVAL_LEN=${16}
MAX_EVAL_LEN=${17}
CONTEXT_LEN=${18}
EVAL_CONTEXT_LEN=${19}
TRAIN_BATCH_SIZE=${20}
EVAL_BATCH_SIZE=${21}
EVAL_NUM_BATCHES=${22}
LR=${23}


conda activate /n/holystore01/LABS/barak_lab/Users/sjelassi/conda/envs/torch-env

python3 main.py --train_task $TRAIN \
		--lr $LR \
	        --model $MODEL \
		--hidden_size $HIDDEN \
		--layers $LAYER \
		--heads $HEADS \
		--epochs $EPOCHS \
		--steps $STEPS \
		--eval_task $EVAL \
		--num_masked_heads ${NUM_MASKED_HEADS} \
		--train_batch_size $TRAIN_BATCH_SIZE\
		--eval_batch_size $EVAL_BATCH_SIZE\
		--eval_num_batches $EVAL_NUM_BATCHES\
		--min_train_len $MIN_TRAIN_LEN\
		--max_train_len $MAX_TRAIN_LEN\
		--min_eval_len $MIN_EVAL_LEN\
		--max_eval_len $MAX_EVAL_LEN\
		--context_len $CONTEXT_LEN\
		--eval_context_len $EVAL_CONTEXT_LEN\
		--n_gram $N_GRAM\
		--length_answer $ANS_LEN\
		--vocab_size $VOCAB_SIZE\
		--state_dim $STATE_DIM\
