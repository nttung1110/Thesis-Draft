#!/bin/sh
#SBATCH -o ./slurm_train_pair_matching/%j.out # STDOUT
python ../script/train_pair_matching.py