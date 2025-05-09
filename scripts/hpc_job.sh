#!/bin/bash

#SBATCH --job-name=bert-training         # Job name
#SBATCH --output=IMDB_model.%j.out   # Name of output file (%j is job ID)
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory
#SBATCH --gres=gpu 					    # Request a GPU
#SBATCH --time=12:00:00                 # Run time (hh:mm:ss)
#SBATCH --partition=scavenge			# Specifies the partition (queue) to submit the job to
#SBATCH --mail-type=BEGIN,END,FAIL		# E-mail when status changes

# Prints the name of the node (computer) where the job is running.
echo "Running on $(hostname):" 

# Command that shows GPU usage and available GPUs
nvidia-smi

conda activate nlp

# Move to folder
cd ~/projects/imdb_bert/scripts

# Run Script
python train.py
