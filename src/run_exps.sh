#!/usr/bin/env bash

#SBATCH --job-name=NTK_exps
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --qos=1week
#SBATCH --time=4-00:00:00

timestamp="$(date "+%H.%M.%S_%d.%m.%Y")"
p="out/${timestamp}"
mkdir -p "${p}"

~/.local/bin/micromamba run -p ./env python src/create_dataset.py -p "${p}/test_dataset.json"

for exp in {0..19}
do
    srun -q 1day --nodes 1 --ntasks 1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --mem-per-cpu=2G ~/.local/bin/micromamba run -p ./env python src/run_exp.py -p "${p}/exp${exp}" "${p}/test_dataset.json" &
done
wait

~/.local/bin/micromamba run -p ./env python src/visualize.py -p "${p}"