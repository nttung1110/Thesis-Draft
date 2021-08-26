#!/bin/sh
#SBATCH -o ./slurm_infer_gt_boxes/%j.out # STDOUT
python ../script/infer_full_pipeline_clone_13_gt_box.py