#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:59:00          # total run time limit (HH:MM:SS)
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_requeue



# _requeue



# _requeue

export MASTER_IP=$(hostname -I | cut -d' ' -f1)

export MASTER_PORT=$(expr 10000 + $(( $RANDOM % 1000 + 1 )) + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * SLURM_GPUS_ON_NODE))


MASTER_IP=$(hostname -I | cut -d' ' -f1)


MODEL=$1
EVAL_TASK=$2
TEXT_ORDER=$3
EVAL_MIN_LEN=$4
EVAL_MAX_LEN=$5
BS_EVAL=$6
NUM_BATCHES=$7


conda activate /n/holystore01/LABS/barak_lab/Users/sjelassi/conda/envs/torch-env

python3 main.py --model $MODEL \
	        --eval_task $EVAL_TASK \
		--text_order $TEXT_ORDER\
		--eval_batch_size $BS_EVAL\
		--eval_num_batches $NUM_BATCHES\
		--min_eval_len $EVAL_MIN_LEN\
		--max_eval_len $EVAL_MAX_LEN\






