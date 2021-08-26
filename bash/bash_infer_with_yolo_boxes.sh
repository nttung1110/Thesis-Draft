#!/bin/sh
#SBATCH -o ./slurm_infer_yolo_boxes/%j.out # STDOUT
python ../script/infer_full_pipeline_detectron.py