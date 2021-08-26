#!/bin/sh
#SBATCH -o ./slurm_train_no_mask/%j.out # STDOUT
python ../script/train_sim_net_no_mask.py