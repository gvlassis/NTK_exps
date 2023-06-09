#!/usr/bin/env bash

timestamp="$(date "+%H.%M.%S_%d.%m.%Y")"
p="out/${timestamp}"
mkdir -p "${p}"

~/.local/bin/micromamba run -p ./env python src/create_dataset.py -p "${p}/test_dataset.json" -N 1 -n 10 -d UniformDataset

for exp in {0..5}; do
    ~/.local/bin/micromamba run -p ./env python src/run_exp.py -p "${p}/exp${exp}" -N 1 -d UniformDataset --max_m_expon 12 "${p}/test_dataset.json" &
done
wait

~/.local/bin/micromamba run -p ./env python src/visualize.py -p "${p}"