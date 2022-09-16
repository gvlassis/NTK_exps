# SPDX-FileCopyrightText: © 2022 Project's authors 
# SPDX-License-Identifier: MIT

import numpy
import torch
import matplotlib.pyplot
import os
import math
import shutil

# Hyperparameters 
N_VALUES = [10, 50, 100]
n_TRAIN_VALUES = [1000, 2500, 5000, 7500, 10000]
n_TEST = 10000
NUM_EXP = 5 # The number of experiments for each (N,n_train) tuple. In each experiment we use a new training dataset but the same testing dataset (V remains the same)
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MAX_TIMES_N = 10
STEP_TIMES_N = 1

# Hyperparameters
BATCH_SIZE = 25
LR = 0.01
MOMENTUM = 0.99

# Stopping criteria 
EPOCHS = 200

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

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class NeuralNetwork(torch.nn.Module):
    def __init__(self, N, m):
        super(NeuralNetwork, self).__init__()
        self.N = N
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

def get_loss(model, dataset):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_function() expects two tensors of the same shape and type
    loss = loss_function(outputs, targets)

    model.train(was_in_training)
    return loss.item()

def train(distribution, test_dataset, n_train, m):
    N = list(distribution.event_shape)[0]

    # Create training curves
    fig, axs = matplotlib.pyplot.subplots(ncols=2, figsize=[20, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'N={N}, n_train={n_train}, m={m}') # One graph for all the experiments 
    
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('train_loss')
    axs[0].grid()
    axs[0].set_yscale('log')

    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('test_loss')
    axs[1].grid()
    axs[1].set_yscale('log')
    
    for exp in range(NUM_EXP):
        train_dataset = SphereDataset(distribution, n_train, test_dataset.V)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        nn = NeuralNetwork(N,m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)

        # Run the experiment
        was_in_training = nn.training
        nn.train(True)

        train_loss_values = numpy.empty(EPOCHS+1) # train_loss_values[i]=The train loss in the BEGINNING of the i-th epoch
        test_loss_values = numpy.empty(EPOCHS+1) # We calculate train/test loss at the beginning and after each epoch (hence EPOCHS+1 times)

        train_loss_values[0] = get_loss(nn, train_dataset)
        test_loss_values[0] = get_loss(nn, test_dataset)

        for epoch in range(EPOCHS):
            if epoch%10==0: print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}')

            for batch, (inputs, targets) in enumerate(train_loader):
                
                # Forward
                outputs = nn(inputs)
                outputs = outputs.reshape(-1)
                loss = loss_function(outputs, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_values[epoch+1] = get_loss(nn, train_dataset)
            test_loss_values[epoch+1] = get_loss(nn, test_dataset)

        axs[0].plot(range(EPOCHS+1), train_loss_values, linestyle='-', marker='o', color='#039be5')
        axs[1].plot(range(EPOCHS+1), test_loss_values, linestyle='-', marker='o', color='#039be5')

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'training_curves/N={0}/n_train={1}/'.format(N, n_train))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'm={m}.pdf')
    matplotlib.pyplot.close(fig)

    nn.train(was_in_training)

    return

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, 'training_curves/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)

for N in N_VALUES:
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N,device=DEVICE), torch.eye(N,device=DEVICE)) # device=DEVICE is essential to get RNG in the GPU 

    test_dataset = SphereDataset(distribution, n_TEST) # We have one V for each N

    m_values = range(N, MAX_TIMES_N*N, STEP_TIMES_N*N)
    for n_train in n_TRAIN_VALUES:
        for m in m_values:
            train(distribution, test_dataset, n_train, m)

    print('Done!')

