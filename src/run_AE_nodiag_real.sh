#!/bin/bash
#$ -N realVal_AE_symmetry_nodiag.log
#$ -cwd
#$ -j y
#$ -P kenprj
#$ -S /bin/bash
#$ -q gpu_long_2080ti
#$ -pe omp 2
module load anaconda3/3.7 
module load cuda10.1/
source /data01/software/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate AIML
python AE_symmetry_nodiag_realVal.py
conda deactivate
conda deactivate