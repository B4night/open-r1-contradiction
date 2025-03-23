#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:4
#SBATCH --time=72:00:00

conda activate open_r1
cd /ibex/user/feic/pjs/open-r1-contradiction
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./recipes/accelerate_configs/zero3.yaml --num_processes=3 ./src/open_r1/grpo.py --config ./recipes/DeepSeek-R1-Distill-Llama-8B/grpo/config_demo.yaml
