#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o ./agg_cosinor_hive_%j.out
#SBATCH -e ./agg_cosinor_hive_%j.out
#SBATCH -J agg_cosinor_hive

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

srun --unbuffered python aggregate_cosinor_estimates_per_hive_position.py --year 2016 --side 0