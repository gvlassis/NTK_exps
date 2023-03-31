# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import argparse
import os
import utils
import json
import numpy
import matplotlib.pyplot

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", metavar="PATH", help="Path of the directory that contains as subdirectories the results of each experiment", dest="p", type=os.path.abspath, default="out")
args=parser.parse_args()

NTK_loss=[]
nn_loss=[]
kern_diff=[]

exps=[os.path.abspath(file) for file in os.scandir(args.p) if file.is_dir()]
for exp in exps:
    with open(exp+f"/NTK_loss.json","r") as NTK_loss_json:
        NTK_loss.append(json.load(NTK_loss_json))
    with open(exp+f"/nn_loss.json","r") as nn_loss_json:
        nn_loss.append(json.load(nn_loss_json))
    with open(exp+f"/kern_diff.json","r") as kern_diff_json:
        kern_diff.append(json.load(kern_diff_json))

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
kern_diff_mean = numpy.mean(kern_diff, axis=1)
kern_diff_std = numpy.std(kern_diff, axis=1)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('kern_diff')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, kern_diff_mean, kern_diff_std, linestyle='-', marker='o', color=utils.INDIGO, ecolor=utils.BLUE, capsize=7)

fig.savefig(args.p+f"/kern_diff.pdf")
matplotlib.pyplot.close(fig)

print("🎨🖌️")


