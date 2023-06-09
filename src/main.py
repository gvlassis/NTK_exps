# SPDX-FileCopyrightText: © 2022 Project's authors 
# SPDX-License-Identifier: MIT

import numpy
import torch
import matplotlib.pyplot
import time
import os
import sklearn.svm
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.metrics
import math
import shutil

# Parameters
N = 8
n_TRAIN = 80
n_TEST = 200
NUM_EXP = 8
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MIN_WIDTH_EXPON = 6
MAX_WIDTH_EXPON = 10

# Hyperparameters
BATCH_SIZE = n_TRAIN
LR = 0.3
MOMENTUM = 0.00

# Stopping criteria 
ALPHA = 4
BETA = -0.2

# Colors
GREEN = '#388E3C'
LIGHT_GREEN = '#8BC34A'
INDIGO = '#3F51B5'
BLUE = '#2196F3'
AMBER = '#FFA000'
LIGHT_AMBER = '#FFCA28'

class SphereDataset(torch.utils.data.Dataset):
    def __init__(self, distribution, n, V = None):
        self.dim = distribution.event_shape
        self.len = n

        if V is None:
            self.V = distribution.sample()
            self.V = self.V / torch.linalg.norm(self.V)
        else:
            self.V = V
        
        self.X = distribution.sample((n,))
        line_norms = torch.linalg.norm(self.X,dim=1)
        line_norms_T = torch.reshape(line_norms,(n,1))
        self.X = self.X/line_norms_T
  
        self.Y = torch.matmul(self.X,self.V)

        self.Y = self.Y >= 0 
        self.Y = self.Y.float()

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class NeuralNetwork(torch.nn.Module):
    def __init__(self, m):
        super(NeuralNetwork, self).__init__()
        self.m = m
        self.hidden_layer = torch.nn.Linear(N, m, bias=False)
        self.output_layer = torch.nn.Linear(m, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1)

    def forward(self, input):
        output = self.hidden_layer(input)
        output = torch.nn.functional.relu(output)
        output = self.output_layer(output)
        output = output*math.sqrt(2/self.m)

        return output

class NeuralNetworkASI(torch.nn.Module):
    def __init__(self, m):
        super(NeuralNetworkASI, self).__init__()
        self.m = m
        
        self.hidden_layer1 = torch.nn.Linear(N, m, bias=False)
        self.output_layer1 = torch.nn.Linear(m, 1, bias=False)
        self.hidden_layer1.weight.data.normal_(mean=0.0, std=1)
        self.output_layer1.weight.data.normal_(mean=0.0, std=1)

        self.hidden_layer2 = torch.nn.Linear(N, m, bias=False)
        self.output_layer2 = torch.nn.Linear(m, 1, bias=False)
        with torch.no_grad(): 
            self.hidden_layer2.weight.copy_(self.hidden_layer1.weight)
            self.output_layer2.weight.copy_(self.output_layer1.weight)

    def forward(self, input):
        output1 = self.hidden_layer1(input)
        output1 = torch.nn.functional.relu(output1)
        output1 = self.output_layer1(output1)
        output1 = output1*math.sqrt(2/self.m)
 
        output2 = self.hidden_layer2(input)
        output2 = torch.nn.functional.relu(output2)
        output2 = self.output_layer2(output2)
        output2 = output2*math.sqrt(2/self.m)

        output = (math.sqrt(2)/2)*(output1 - output2)

        return output

def get_loss(model, dataset):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_function() expects two tensors of the same shape and type
    loss = loss_function(outputs, targets)

    model.train(was_in_training)
    return loss.item()
 
def k0(u):
    return (1/math.pi)*(math.pi-math.acos(u))

def k1(u):
    return (1/math.pi)*(u*(math.pi-math.acos(u))+math.sqrt(1-u**2))

def k(u):
    return u*k0(u)+k1(u)

def K(a,b):
    norma = numpy.linalg.norm(a, 2)
    normb = numpy.linalg.norm(b, 2)
    normprod = norma*normb
    inprod = numpy.dot(a,b)
    u = inprod/normprod

    # Fix values outside of [-1,1] due to computation inaccuracies
    if(u<-1): u=-1
    elif(u>1): u=1

    return normprod*k(u)

def phi(model, input):
    # Forward
    output = model(input)

    # Backward
    model.zero_grad()
    output.backward()
    
    temp = torch.tensor([])

    for param in model.parameters():
        temp=torch.cat((temp, param.grad.reshape(-1)))

    return temp

def Kw(model,a,b):
    return torch.dot(phi(model,a),phi(model,b))

def Kw_matrix(inputs,model):

    out = torch.empty((inputs.shape[0],inputs.shape[0]))

    for i, a in enumerate(inputs):
        for j, b in enumerate(inputs):
            out[i][j] = Kw(model,a,b)
    
    return out

def has_converged(a):
    y1 = math.log(a[int(len(a)/ALPHA)],10)
    y2 = math.log(a[-1],10)
    dy = y2-y1
    range_y = math.log(max(a),10)-math.log(min(a),10)
    if range_y == 0: slope=0
    else: slope = dy/(range_y*(1-1/ALPHA))
    
    print(f'slope={slope}')
    if slope > BETA:
        return True

    return False

def train(model, optimizer, train_dataset, m, exp, is_frozen):
    was_in_training = model.training
    model.train(True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    (train_inputs, train_targets) = train_dataset[:]

    epoch_values = [] # We do not know when the training will finish
    train_loss_values = [] # train_loss_values[i]=The train loss in the BEGINNING of the i-th epoch

    epoch = 0
    while(True):
        
        epoch_values.append(epoch)
        train_loss_values.append(get_loss(model, train_dataset))

        if (epoch+1)%ALPHA==0:
            print(f'm={m}, exp={exp}, is_frozen={is_frozen}, epoch={epoch}, train_loss={train_loss_values[-1]}, ',end="")
            if has_converged(train_loss_values):
                break

	    # If we do not have convergence...
        for batch, (inputs, targets) in enumerate(train_loader):
            
            # Forward
            outputs = model(inputs)
            outputs = outputs.reshape(-1)
            loss = loss_function(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch += 1

    # Create training curves
    fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'm={m}, exp={exp}, is_frozen={is_frozen}')

    axs.plot(epoch_values, train_loss_values, linestyle='-', marker='o', color=BLUE)
    axs.set_xlabel('epoch')
    axs.set_ylabel('train_loss')
    axs.grid()
    axs.set_yscale('log')

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, '../output/training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp},is_frozen={is_frozen}.pdf')
    matplotlib.pyplot.close(fig)

    model.train(was_in_training)

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, '../output/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)

distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N,device=DEVICE), torch.eye(N,device=DEVICE))

test_dataset = SphereDataset(distribution, n_TEST)
(test_inputs, test_targets) = test_dataset[:]

NTK_loss = numpy.empty(NUM_EXP)
m_exponents = range(MIN_WIDTH_EXPON, MAX_WIDTH_EXPON+1)
m_values = [2**exp for exp in m_exponents]
nn_loss = numpy.empty([NUM_EXP, len(m_values)])
kern_diff = numpy.empty([NUM_EXP, len(m_values)])
nn_loss_frozen = numpy.empty([NUM_EXP, len(m_values)])
for exp in range(NUM_EXP):
    # Sample new train_dataset
    train_dataset = SphereDataset(distribution, n_TRAIN, test_dataset.V)

    # Train the NTK
    (train_inputs, train_targets) = train_dataset[:]

    NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
    print(f'Training NTK, exp={exp}')
    NTK.fit(train_inputs.cpu().numpy(), train_targets.cpu().numpy()) # .numpy() only takes tensor in CPU
    print(f'Infering NTK, exp={exp}')
    test_outputs_NTK = NTK.predict(test_inputs.cpu().numpy()) # .numpy() only takes tensor in CPU
    NTK_loss[exp] = sklearn.metrics.mean_squared_error(test_targets.cpu().numpy(), test_outputs_NTK) # .numpy() only takes tensor in CPU

    # Train the nns
    for m_index, m in enumerate(m_values):
        nn = NeuralNetworkASI(m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)

        Kw0_matrix = Kw_matrix(train_inputs, nn)
        
        # Train the nn
        train(nn, optimizer, train_dataset, m, exp, False)

        nn_loss[exp,m_index] = get_loss(nn, test_dataset)

        Kwconv_matrix = Kw_matrix(train_inputs,nn)
        kern_diff[exp,m_index] = torch.linalg.matrix_norm(Kwconv_matrix-Kw0_matrix, ord=2)

        # Frozen nn
        nn_frozen = NeuralNetworkASI(m)
        nn_frozen.hidden_layer1.requires_grad_(False)
        nn_frozen.hidden_layer2.requires_grad_(False)
        nn_frozen.to(DEVICE)

        # Set up the optimizer for nn_frozen
        optimizer = torch.optim.SGD(nn_frozen.parameters(), lr=LR, momentum=MOMENTUM)
        
        # Train nn_frozen
        train(nn_frozen, optimizer, train_dataset, m, exp, True)

        nn_loss_frozen[exp,m_index] = get_loss(nn_frozen, test_dataset)

# l2_loss plot
NTK_loss_mean = numpy.mean(NTK_loss)
NTK_loss_std = numpy.std(NTK_loss)
nn_loss_mean = numpy.mean(nn_loss, axis=0)
nn_loss_std = numpy.std(nn_loss, axis=0)
nn_loss_frozen_mean = numpy.mean(nn_loss_frozen, axis=0)
nn_loss_frozen_std = numpy.std(nn_loss_frozen, axis=0)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('l2_loss')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, NTK_loss_mean*numpy.ones(len(m_values)), NTK_loss_std*numpy.ones(len(m_values)), linestyle='-', marker='o', color=GREEN, ecolor=LIGHT_GREEN, capsize=7)
axs.errorbar(m_values, nn_loss_mean, nn_loss_std, linestyle='-', marker='o', color=INDIGO, ecolor=BLUE, capsize=7)
axs.errorbar(m_values, nn_loss_frozen_mean, nn_loss_frozen_std, linestyle='-', marker='o', color=AMBER, ecolor=LIGHT_AMBER, capsize=7)

script_dir = os.path.dirname(__file__)
fig.savefig(script_dir+'/../output/l2_loss.pdf')
matplotlib.pyplot.close(fig)

# kern_diff plot
kern_diff_mean = numpy.mean(kern_diff, axis=1)
kern_diff_std = numpy.std(kern_diff, axis=1)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('kern_diff')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, kern_diff_mean, kern_diff_std, linestyle='-', marker='o', color=INDIGO, ecolor=BLUE, capsize=7)

script_dir = os.path.dirname(__file__)
fig.savefig(script_dir+'/../output/kern_diff.pdf')
matplotlib.pyplot.close(fig)

print('🧪')