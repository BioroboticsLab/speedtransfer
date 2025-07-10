#!/bin/bash -l
#SBATCH -D ./

#SBATCH --array=0-199

#SBATCH -o ./output_%A_%a.out
#SBATCH -e ./output_%A_%a.out

#SBATCH -J read_velocities

#SBATCH --qos='standard'

### TIME LIMIT.
#SBATCH --time=0-02:00:00
#SBATCH --signal=USR1@300

### CHANGE THIS TO YOUR EMAIL ADRESS.
#SBATCH --mail-type=all
#SBATCH --mail-user=yourname@mail.com

### ADJUST MEMORY IF NEEDED.
#SBATCH --mem=32G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

### REPLACE WITH YOUR USERNAME HERE.
source /home/YOUR_USERNAME/.bashrc
source activate speedtransfer

srun --unbuffered python read_velocities_from_db.py \
     --date_from 20 08 2019 \
     --date_to 14 09 2019 \
     --batch $SLURM_ARRAY_TASK_ID

srun --unbuffered python read_velocities_from_db.py \
     --date_from 01 08 2016 \
     --date_to 25 08 2016 \
     --batch $SLURM_ARRAY_TASK_ID