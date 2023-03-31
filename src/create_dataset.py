# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import argparse
import os
import torch
import utils
import json
import datasets

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", "--input_dimension", metavar="INT", help="Input dimension", dest="N", type=int, default=8)
parser.add_argument("-n", "--num_points", metavar="INT", help="Number of points", dest="n", type=int, default=10000)
parser.add_argument("-p", "--path", metavar="PATH", help="Path of the JSON file", dest="p", type=os.path.abspath, default="dataset.json")
parser.add_argument("-d", "--dataset", metavar="DATASET", help="Class name of the dataset", dest="d", type=utils.string_to_dataset_class, default="DiscreteSphereDataset")
parser.add_argument("-D", "--device", metavar="DEVICE", help="Device used for the random number generation", dest="D", type=torch.device, default="cpu")
args=parser.parse_args()

distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.N,device=args.D), torch.eye(args.N,device=args.D))
dataset = args.d.create_without_V(distribution, args.n)

os.makedirs(os.path.dirname(args.p), exist_ok=True)
with open(args.p,"w") as dataset_json:
    json.dump(dataset, dataset_json, cls=datasets.HyperplaneDatasetEncoder, indent=4)