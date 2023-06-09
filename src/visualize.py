# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import argparse
import os
import utils
import json
import numpy
import matplotlib.pyplot
import datasets

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", metavar="PATH", help="Path of the directory that contains as subdirectories the results of each experiment", dest="p", type=os.path.abspath, default="out")
args=parser.parse_args()

NTK_loss=[]
nn_loss=[]
kern_diff=[]
v_maxes = []

exps=[os.path.abspath(file) for file in os.scandir(args.p) if file.is_dir()]
for exp in exps:
    with open(exp+f"/NTK_loss.json","r") as NTK_loss_json:
        NTK_loss.append(json.load(NTK_loss_json))
    with open(exp+f"/nn_loss.json","r") as nn_loss_json:
        nn_loss.append(json.load(nn_loss_json))
    with open(exp+f"/kern_diff.json","r") as kern_diff_json:
        kern_diff.append(json.load(kern_diff_json))
    with open(exp+f"/v_maxes.json","r") as v_maxes_json:
        v_maxes.append(json.load(v_maxes_json))

with open(exp+f"/m_values.json","r") as m_values_json:
        m_values=json.load(m_values_json)

# l2_loss plot
NTK_loss_mean = numpy.mean(NTK_loss)
NTK_loss_std = numpy.std(NTK_loss)
nn_loss_mean = numpy.mean(nn_loss, axis=0)
nn_loss_std = numpy.std(nn_loss, axis=0)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('l2_loss')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, NTK_loss_mean*numpy.ones(len(m_values)), NTK_loss_std*numpy.ones(len(m_values)), linestyle='-', marker='o', color=utils.GREEN, ecolor=utils.LIGHT_GREEN, capsize=7)
axs.errorbar(m_values, nn_loss_mean, nn_loss_std, linestyle='-', marker='o', color=utils.INDIGO, ecolor=utils.BLUE, capsize=7)

fig.savefig(args.p+f"/l2_loss.pdf")
matplotlib.pyplot.close(fig)

# kern_diff plot
kern_diff_mean = numpy.mean(kern_diff, axis=0)
kern_diff_std = numpy.std(kern_diff, axis=0)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('kern_diff')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, kern_diff_mean, kern_diff_std, linestyle='-', marker='o', color=utils.INDIGO, ecolor=utils.BLUE, capsize=7)

fig.savefig(args.p+f"/kern_diff.pdf")
matplotlib.pyplot.close(fig)

# v_maxes plot
v_maxes_mean = numpy.mean(v_maxes, axis=0)
v_maxes_std = numpy.std(v_maxes, axis=0)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('x')
axs.set_ylabel('eigenfunction')
axs.grid()

with open(f"{args.p}/test_dataset.json","r") as test_dataset_json:
    test_dataset=json.load(test_dataset_json, object_hook=datasets.decode_hyperplane_dataset)
(test_inputs, test_targets) = test_dataset[:]

for i, v_max_mean in enumerate(v_maxes_mean):
    col=numpy.random.rand(3,)
    axs.errorbar(numpy.ravel(test_inputs), v_max_mean, v_maxes_std[i], fmt='o', color=col, ecolor=col, capsize=7)

fig.savefig(args.p+f"/v_maxes.pdf")
matplotlib.pyplot.close(fig)

print("🎨🖌️")


