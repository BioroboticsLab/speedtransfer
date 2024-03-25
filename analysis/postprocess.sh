#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o ./postprocess_%j.out
#SBATCH -e ./postprocess_%j.out
#SBATCH -J postprocess

#SBATCH --qos='standard'

### TIME LIMIT
#SBATCH --time=0-32:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=jmellert@fu-berlin.de

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

source /home/juliam98/.bashrc
source activate speedtransfer

srun --unbuffered python social_network_interaction_tree.py --postprocess