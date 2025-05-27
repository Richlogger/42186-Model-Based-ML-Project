#!/bin/bash
#BSUB -J run_model4_new
#BSUB -o output_%J.out
#BSUB -e error_%J.err
#BSUB -n 8                     # Number of cores
#BSUB -R "rusage[mem=64GB]"   # Memory requirement
#BSUB -W 4:00                 # Walltime (e.g., 4 hours)


# Activate environment
source ~/course_42186/bin/activate

# Move to your project directory
cd /work3/s214790/42186-Model-Based-ML-Project/


# Train and evaluate the HDPMM model on the full dataset
papermill model4_new.ipynb output_model4_new_${LSB_JOBID}.ipynb
