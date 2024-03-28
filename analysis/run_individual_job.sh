#!/bin/bash -l
#SBATCH -D ./

### SPECIFY WHERE THE OUTPUT WILL BE SAVED TO.
#SBATCH -o ./output_%j.out
#SBATCH -e ./output_%j.out

#SBATCH -J NAME_OF_JOB

#SBATCH --qos='standard'

### TIME LIMIT.
#SBATCH --time=0-01:00:00
#SBATCH --signal=USR1@300

### CHANGE THIS TO YOUR EMAIL ADRESS.
#SBATCH --mail-type=all
#SBATCH --mail-user=yourname@mail.com

### ADJUST MEMORY IF NEEDED.
#SBATCH --mem=50G

### NO NEED TO CHNAGE THIS.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1


### REPLACE HERE.
source /home/YOUR_USERNAME/.bashrc
source activate NAME_OF_CONDA_ENVIRONMENT

# REPLACE WITH THE SCRIPT AND ARGUMENTS YOU WOULD LIKE TO RUN.
srun --unbuffered python point_of_interaction.py --year 2019 --side 0 --focal 0