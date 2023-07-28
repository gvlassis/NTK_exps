# SPDX-FileCopyrightText: © 2022 Project's authors 
# SPDX-License-Identifier: MIT

import numpy
import torch
import matplotlib.figure
import time
import os
import sklearn.svm
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.metrics
import math
import shutil
import numpy 

# Parameters
N = 15
m = 1000
n_TRAIN_values = [32, 64, 128, 256, 512]
n_TEST = 10000
κ = 3
r = 1/(3*κ-1)
δ = 3/(3*κ-1)
CENTERS_X = (-1/2+r+δ*torch.arange(κ)).repeat(κ,1)
CENTERS_Y = (-1/2+r+δ*torch.arange(κ)).repeat(κ,1).T
CENTERS = torch.stack((CENTERS_X,CENTERS_Y),dim=2)
NUM_EXP = 10
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()

# Hyperparameters
LR = 0.5
MOMENTUM = 0.00

# Stopping criteria 
ITERATIONS = 50000

class ChizatDataset(torch.utils.data.Dataset): 
    def __init__(self, categorical_distribution, radius_uniform_distribution, angle_uniform_distribution, noise_uniform_distribution, bernoulli_distribution, n, N, V = None):
        self.len = n 
        self.dim = N
        self.κ = categorical_distribution.probs.shape[0]

        if V is None:
            self.V = (bernoulli_distribution.sample([self.κ,self.κ])-0.5)*2
        else:
            self.V = V
        
        # datapoint->disk
        D = categorical_distribution.sample([n,2])
        # datapoint->center
        C = CENTERS[D[:,0],D[:,1]]
        R = radius_uniform_distribution.sample([n])
        Φ = angle_uniform_distribution.sample([n])

        X_x = C[:,0] + R*torch.cos(Φ)
        X_y = C[:,1] + R*torch.sin(Φ)
        X_xy = torch.stack((X_x,X_y),dim=1)
        X_noise = noise_uniform_distribution.sample([n,N-2])

        self.X = torch.cat((X_xy,X_noise),dim=1)

        self.Y = self.V[D[:,0],D[:,1]]
        
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class ChizatDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, chizat_dataset):
        self.len = chizat_dataset.len
        self.dim = chizat_dataset.dim
        self.κ = chizat_dataset.κ
        self.V = chizat_dataset.V

        X_1 = torch.full([chizat_dataset.len,1],1)
        self.X = torch.cat((chizat_dataset.X,X_1),dim=1)

        self.Y = chizat_dataset.Y
        
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class NeuralNetworkASI(torch.nn.Module):
    def __init__(self, m):
        super(NeuralNetworkASI, self).__init__()
        self.m = m
        
        self.hidden_layer1 = torch.nn.Linear(N+1, m, bias=False)
        self.output_layer1 = torch.nn.Linear(m, 1, bias=False)
        self.hidden_layer1.weight.data.normal_(mean=0.0, std=1)
        self.output_layer1.weight.data.normal_(mean=0.0, std=1)

        self.hidden_layer2 = torch.nn.Linear(N+1, m, bias=False)
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

    # Exact solution
    def train_output(self, X, Y):
        Y = torch.reshape(Y, (-1,1))

        Z_1 = self.hidden_layer1(X)
        Z_1 = torch.nn.functional.relu(Z_1)

        Z_2 = self.hidden_layer2(X)
        Z_2 = torch.nn.functional.relu(Z_2)

        Z_ = torch.cat( (Z_1/math.sqrt(self.m), -Z_2/math.sqrt(self.m)), 1)

        # It is always preferred to use lstsq() when possible, as it is faster and more numerically stable than computing the pseudoinverse explicitly.
        V = torch.linalg.lstsq(Z_,Y).solution.T

        with torch.no_grad(): 
            self.output_layer1.weight.copy_(V[0,:self.m])
            self.output_layer2.weight.copy_(V[0,self.m:])

def get_loss(model, dataset):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_function() expects two tensors of the same shape and type
    loss = loss_function(outputs, targets)

    model.train(was_in_training)
    return loss.item()

def get_error(model, dataset):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_function() expects two tensors of the same shape and type
    predictions = (outputs > 0)*2-1
    errors = torch.sum(predictions != targets)
    error = errors/len(targets)

    model.train(was_in_training)
    return error.item()
 
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

def K_RF(a,b):
    norma = numpy.linalg.norm(a, 2)
    normb = numpy.linalg.norm(b, 2)
    normprod = norma*normb
    inprod = numpy.dot(a,b)
    u = inprod/normprod

    # Fix values outside of [-1,1] due to computation inaccuracies
    if(u<-1): u=-1
    elif(u>1): u=1

    return (normprod/2)*k1(u)

def train(model, optimizer, train_dataset, exp, n_TRAIN):
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

        print(f'exp={exp}, n_TRAIN={n_TRAIN}, epoch={epoch}, train_loss={train_loss_values[-1]}')

        if epoch>ITERATIONS:
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
    fig = matplotlib.figure.Figure()
    fig.suptitle(f'm={m}, exp={exp}')
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0],xlabel="epoch",ylabel="train_loss")   
    ax.plot(epoch_values, train_loss_values, marker='o')
    ax.set_yscale('log')
    ax.grid()

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, '../output/training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')

    model.train(was_in_training)

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, '../output/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)

probs = torch.full([κ],1/κ)
categorical_distribution = torch.distributions.categorical.Categorical(probs)
radius_uniform_distribution = torch.distributions.uniform.Uniform(0, r)
angle_uniform_distribution = torch.distributions.uniform.Uniform(0, 2*math.pi)
noise_uniform_distribution = torch.distributions.uniform.Uniform(-1/2, 1/2)
bernoulli_distribution = torch.distributions.bernoulli.Bernoulli(0.5)

test_dataset = ChizatDataset(categorical_distribution, radius_uniform_distribution, angle_uniform_distribution, noise_uniform_distribution, bernoulli_distribution, n_TEST, N)
test_dataset_augmented = ChizatDatasetAugmented(test_dataset)
(test_inputs, test_targets) = test_dataset[:]
(test_inputs_augmented, test_targets_augmented) = test_dataset_augmented[:]

NTK_error = numpy.empty([NUM_EXP, len(n_TRAIN_values)])
RFK_error = numpy.empty([NUM_EXP, len(n_TRAIN_values)])
nn_error = numpy.empty([NUM_EXP, len(n_TRAIN_values)])
nn_error_frozen = numpy.empty([NUM_EXP, len(n_TRAIN_values)])
for exp in range(NUM_EXP):
    for i, n_TRAIN in enumerate(n_TRAIN_values):
        BATCH_SIZE = n_TRAIN
        
        # Sample new train_dataset
        train_dataset = ChizatDataset(categorical_distribution, radius_uniform_distribution, angle_uniform_distribution, noise_uniform_distribution, bernoulli_distribution, n_TRAIN, N, test_dataset.V)
        train_dataset_augmented = ChizatDatasetAugmented(train_dataset)

        # Train the NTK
        (train_inputs, train_targets) = train_dataset[:]
        (train_inputs_augmented, train_targets_augmented) = train_dataset_augmented[:]

        NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
        print(f'Training NTK, exp={exp}, n_TRAIN={n_TRAIN}')
        NTK.fit(train_inputs_augmented.cpu().numpy(), train_targets_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
        print(f'Infering NTK, exp={exp}, n_TRAIN={n_TRAIN}')
        test_outputs_NTK = NTK.predict(test_inputs_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
        test_predictions_NTK = (test_outputs_NTK > 0)*2-1
        test_errors_NTK = numpy.sum(test_predictions_NTK != test_targets_augmented.cpu().numpy())
        NTK_error[exp,i] = test_errors_NTK/len(test_targets_augmented)

        # Train the RFK
        RFK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K_RF)
        print(f'Training RFK, exp={exp}, n_TRAIN={n_TRAIN}')
        RFK.fit(train_inputs_augmented.cpu().numpy(), train_targets_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
        print(f'Infering RFK, exp={exp}, n_TRAIN={n_TRAIN}')
        test_outputs_RFK = RFK.predict(test_inputs_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
        test_predictions_RFK = (test_outputs_RFK > 0)*2-1
        test_errors_RFK = numpy.sum(test_predictions_RFK != test_targets_augmented.cpu().numpy())
        RFK_error[exp,i] = test_errors_RFK/len(test_targets_augmented)

        
        nn = NeuralNetworkASI(m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)
        
        # Train the nn
        train(nn, optimizer, train_dataset_augmented, exp, n_TRAIN)

        nn_error[exp,i] = get_error(nn, test_dataset_augmented)

        # Frozen nn
        nn_frozen = NeuralNetworkASI(m)
        nn_frozen.to(DEVICE)

        nn_frozen.train_output(train_inputs_augmented, train_targets_augmented)

        nn_error_frozen[exp,i] = get_error(nn_frozen, test_dataset_augmented)

# error plot
NTK_error_mean = numpy.mean(NTK_error, axis=0)
NTK_error_std = numpy.std(NTK_error, axis=0)
RFK_error_mean = numpy.mean(RFK_error, axis=0)
RFK_error_std = numpy.std(RFK_error, axis=0)
nn_error_mean = numpy.mean(nn_error, axis=0)
nn_error_std = numpy.std(nn_error, axis=0)
nn_error_frozen_mean = numpy.mean(nn_error_frozen, axis=0)
nn_error_frozen_std = numpy.std(nn_error_frozen, axis=0)

fig = matplotlib.figure.Figure()
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0],xlabel="n_TRAIN",ylabel="error")
ax.grid()

ax.plot(n_TRAIN_values,NTK_error_mean,marker="o",label="NTK")
ax.fill_between(n_TRAIN_values,NTK_error_mean-NTK_error_std,NTK_error_mean+NTK_error_std,alpha=3/8)
ax.plot(n_TRAIN_values,RFK_error_mean,marker="o",label="RFK")
ax.fill_between(n_TRAIN_values,RFK_error_mean-RFK_error_std,RFK_error_mean+RFK_error_std,alpha=3/8)
ax.plot(n_TRAIN_values,nn_error_mean,marker="o",label="NN")
ax.fill_between(n_TRAIN_values,nn_error_mean-nn_error_std,nn_error_mean+nn_error_std,alpha=3/8)
ax.plot(n_TRAIN_values,nn_error_frozen_mean,marker="o",label="NN_frozen")
ax.fill_between(n_TRAIN_values,nn_error_frozen_mean-nn_error_frozen_std,nn_error_frozen_mean+nn_error_frozen_std,alpha=3/8)
ax.legend()

script_dir = os.path.dirname(__file__)
fig.savefig(script_dir+'/../output/error.pdf')

print('🧪')