# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import argparse
import os
import torch
import utils
import json
import datasets
import sklearn.svm
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.metrics
import numpy
import models

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(metavar="PATH", help="Path of the JSON file with the test dataset", dest="test_dataset_path", type=os.path.abspath)
parser.add_argument("-N", "--input_dimension", metavar="INT", help="Input dimension", dest="N", type=int, default=8)
parser.add_argument("-n", "--num_points", metavar="INT", help="Number of points in each train dataset", dest="n", type=int, default=100)
parser.add_argument("-p", "--path", metavar="PATH", help="Path of the directory of the results of this experiment", dest="p", type=os.path.abspath, default="exp")
parser.add_argument("-d", "--dataset", metavar="DATASET", help="Class name of the dataset", dest="d", type=utils.string_to_dataset_class, default="DiscreteSphereDataset")
parser.add_argument("-D", "--device", metavar="DEVICE", help="Device used for the random number generation and the training of the neural networks", dest="D", type=torch.device, default="cpu")
parser.add_argument("--min_m_expon", metavar="INT", help="Minimum width exponent", type=int, default=6)
parser.add_argument("--max_m_expon", metavar="INT", help="Maximum width exponent", type=int, default=14)
parser.add_argument("--batch_size", metavar="INT", help="Batch size (0 corresponds to vanilla/non-stochastic gradient descent)", type=int, default=0)
parser.add_argument("--lr", metavar="FLOAT", help="Learning rate used for (stochastic) gradient descent", type=float, default=0.3)
parser.add_argument("--momentum", metavar="FLOAT", help="Momentum used for (stochastic) gradient descent", type=float, default=0.0)
parser.add_argument("-a", "--alpha", metavar="FLOAT", help="The alpha parameter for the convergence criterion", dest="a", type=float, default=4)
parser.add_argument("-b", "--beta", metavar="FLOAT", help="The beta parameter for the convergence criterion", dest="b", type=float, default=-0.2)
args=parser.parse_args()

if args.batch_size==0: args.batch_size=args.n
print(args.batch_size)

with open(args.test_dataset_path,"r") as test_dataset_json:
    test_dataset=json.load(test_dataset_json, object_hook=datasets.decode_hyperplane_dataset)
(test_inputs, test_targets) = test_dataset[:]

distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.N,device=args.D), torch.eye(args.N,device=args.D))
train_dataset = args.d(distribution, args.n, test_dataset.V)
(train_inputs, train_targets) = train_dataset[:]

# Train the NTK
NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=models.K)
print(f"Training the NTK")
NTK.fit(train_inputs.cpu().numpy(), train_targets.cpu().numpy()) # .numpy() only takes tensor in CPU
print(f"Infering the NTK")
test_outputs_NTK = NTK.predict(test_inputs.cpu().numpy()) # .numpy() only takes tensor in CPU
NTK_loss = sklearn.metrics.mean_squared_error(test_targets.cpu().numpy(), test_outputs_NTK) # .numpy() only takes tensor in CPU

# Train the neural networks
m_exponents = range(args.min_m_expon, args.max_m_expon+1)
m_values = [2**exp for exp in m_exponents]
nn_loss=[]
kern_diff=[]
for m_index, m in enumerate(m_values):
    nn = models.NeuralNetworkASI(args.N, m)
    nn.to(args.D)

    # Set up the optimizer for the nn
    optimizer = torch.optim.SGD(nn.parameters(), lr=args.lr, momentum=args.momentum)

    Kw0_matrix = models.Kw_matrix(train_inputs, nn)
    
    # Train the nn
    models.train(nn, optimizer, train_dataset, args.batch_size, args.a, args.b, args.p)

    nn_loss.append(models.get_loss(nn, test_dataset))

    Kwconv_matrix = models.Kw_matrix(train_inputs,nn)
    kern_diff.append(torch.linalg.matrix_norm(Kwconv_matrix-Kw0_matrix, ord=2).item())

# Serialization
os.makedirs(args.p, exist_ok=True)
with open(args.p+f"/m_values.json","w") as m_values_json:
    json.dump(m_values, m_values_json, indent=4)
with open(args.p+f"/NTK_loss.json","w") as NTK_loss_json:
    json.dump(NTK_loss, NTK_loss_json, indent=4)
with open(args.p+f"/nn_loss.json","w") as nn_loss_json:
    json.dump(nn_loss, nn_loss_json, indent=4)
with open(args.p+f"/kern_diff.json","w") as kern_diff_json:
    json.dump(kern_diff, kern_diff_json, indent=4)

print("🥼🧪")