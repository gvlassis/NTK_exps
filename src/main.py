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
N = 8
n_TRAIN = 80
n_TEST = 10000
κ = 3
r = 1/(3*κ-1)
δ = 3/(3*κ-1)
CENTERS_X = (-1/2+r+δ*torch.arange(κ)).repeat(κ,1)
CENTERS_Y = (-1/2+r+δ*torch.arange(κ)).repeat(κ,1).T
CENTERS = torch.stack((CENTERS_X,CENTERS_Y),dim=2)
NUM_EXP = 2
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MIN_WIDTH_EXPON = 7
MAX_WIDTH_EXPON = 8 

# Hyperparameters
BATCH_SIZE = n_TRAIN
LR = 1
MOMENTUM = 0.95

# Stopping criteria 
ALPHA = 4
BETA = -0.2

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

def has_converged(a):
    y1 = math.log(a[int(len(a)/ALPHA)],10)
    y2 = math.log(a[-1],10)
    dy = y2-y1
    range_y = math.log(max(a),10)-math.log(min(a),10)
    if range_y == 0: slope=0
    else: slope = dy/(range_y*(1-1/ALPHA))
    
    print(f'slope={slope}')
    if slope > BETA and a[-1] < 3e-2:
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
    fig = matplotlib.figure.Figure()
    fig.suptitle(f'm={m}, exp={exp}, is_frozen={is_frozen}')
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0],xlabel="epoch",ylabel="train_loss")   
    ax.plot(epoch_values, train_loss_values, marker='o')
    ax.set_yscale('log')
    ax.grid()

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, '../output/training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp},is_frozen={is_frozen}.pdf')

    model.train(was_in_training)

def plot_learned_function(x,y,z,loss,name):
    fig = matplotlib.figure.Figure()
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0],title="loss=%.4f" % (loss),xlabel="x",ylabel="y")  
    ax.grid()
    ax.scatter(x, y, c=z)
    script_dir = os.path.dirname(__file__)
    fig.savefig(script_dir+'/../output/'+name+'.pdf')

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

NTK_loss = numpy.empty(NUM_EXP)
RFK_loss = numpy.empty(NUM_EXP)
m_exponents = range(MIN_WIDTH_EXPON, MAX_WIDTH_EXPON+1)
m_values = [2**exp for exp in m_exponents]
nn_loss = numpy.empty([NUM_EXP, len(m_values)])
nn_loss_frozen = numpy.empty([NUM_EXP, len(m_values)])
for exp in range(NUM_EXP):
    # Sample new train_dataset
    train_dataset = ChizatDataset(categorical_distribution, radius_uniform_distribution, angle_uniform_distribution, noise_uniform_distribution, bernoulli_distribution, n_TRAIN, N, test_dataset.V)
    train_dataset_augmented = ChizatDatasetAugmented(train_dataset)

    # Train the NTK
    (train_inputs, train_targets) = train_dataset[:]
    (train_inputs_augmented, train_targets_augmented) = train_dataset_augmented[:]

    NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
    print(f'Training NTK, exp={exp}')
    NTK.fit(train_inputs_augmented.cpu().numpy(), train_targets_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
    print(f'Infering NTK, exp={exp}')
    test_outputs_NTK = NTK.predict(test_inputs_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
    NTK_loss[exp] = sklearn.metrics.mean_squared_error(test_targets_augmented.cpu().numpy(), test_outputs_NTK) # .numpy() only takes tensor in CPU

    # Train the RFK
    RFK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K_RF)
    print(f'Training RFK, exp={exp}')
    RFK.fit(train_inputs_augmented.cpu().numpy(), train_targets_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
    print(f'Infering RFK, exp={exp}')
    test_outputs_RFK = RFK.predict(test_inputs_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
    RFK_loss[exp] = sklearn.metrics.mean_squared_error(test_targets_augmented.cpu().numpy(), test_outputs_RFK) # .numpy() only takes tensor in CPU

    # Train the nns
    for m_index, m in enumerate(m_values):
        nn = NeuralNetworkASI(m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)
        
        # Train the nn
        train(nn, optimizer, train_dataset_augmented, m, exp, False)

        nn_loss[exp,m_index] = get_loss(nn, test_dataset_augmented)

        # Frozen nn
        nn_frozen = NeuralNetworkASI(m)
        nn_frozen.to(DEVICE)

        nn_frozen.train_output(train_inputs_augmented, train_targets_augmented)

        nn_loss_frozen[exp,m_index] = get_loss(nn_frozen, test_dataset_augmented)

# l2_loss plot
NTK_loss_mean = numpy.mean(NTK_loss)
NTK_loss_std = numpy.std(NTK_loss)
RFK_loss_mean = numpy.mean(RFK_loss)
RFK_loss_std = numpy.std(RFK_loss)
nn_loss_mean = numpy.mean(nn_loss, axis=0)
nn_loss_std = numpy.std(nn_loss, axis=0)
nn_loss_frozen_mean = numpy.mean(nn_loss_frozen, axis=0)
nn_loss_frozen_std = numpy.std(nn_loss_frozen, axis=0)

fig = matplotlib.figure.Figure()
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0],xlabel="m",ylabel="l2_loss")
ax.set_xscale('log', base=2)
ax.grid()

ax.plot(m_values,numpy.full((len(m_values)),NTK_loss_mean),marker="o",label="NTK")
ax.fill_between(m_values,numpy.full((len(m_values)),NTK_loss_mean-NTK_loss_std),numpy.full((len(m_values)),NTK_loss_mean+NTK_loss_std),alpha=3/8)
ax.plot(m_values,numpy.full((len(m_values)),RFK_loss_mean),marker="o",label="RFK")
ax.fill_between(m_values,numpy.full((len(m_values)),RFK_loss_mean-RFK_loss_std),numpy.full((len(m_values)),RFK_loss_mean+RFK_loss_std),alpha=3/8)
ax.plot(m_values,nn_loss_mean,marker="o",label="NN")
ax.fill_between(m_values,nn_loss_mean-nn_loss_std,nn_loss_mean+nn_loss_std,alpha=3/8)
ax.plot(m_values,nn_loss_frozen_mean,marker="o",label="NN_frozen")
ax.fill_between(m_values,nn_loss_frozen_mean-nn_loss_frozen_std,nn_loss_frozen_mean+nn_loss_frozen_std,alpha=3/8)
ax.legend()

script_dir = os.path.dirname(__file__)
fig.savefig(script_dir+'/../output/l2_loss.pdf')

# Plots for debugging
train_outputs_NTK = NTK.predict(train_inputs_augmented.cpu().numpy()) # .numpy() only takes tensor in CPU
NTK_train_loss = sklearn.metrics.mean_squared_error(train_targets_augmented.cpu().numpy(), train_outputs_NTK) # .numpy() only takes tensor in CPU

NTK_no_bias = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
print(f'Training NTK_no_bias')
NTK_no_bias.fit(train_inputs.cpu().numpy(), train_targets.cpu().numpy()) # .numpy() only takes tensor in CPU
print(f'Infering NTK_no_bias')
test_outputs_NTK_no_bias = NTK_no_bias.predict(test_inputs.cpu().numpy()) # .numpy() only takes tensor in CPU
NTK_no_bias_loss = sklearn.metrics.mean_squared_error(test_targets.cpu().numpy(), test_outputs_NTK_no_bias) # .numpy() only takes tensor in CPU
train_outputs_NTK_no_bias = NTK_no_bias.predict(train_inputs.cpu().numpy()) # .numpy() only takes tensor in CPU
NTK_no_bias_train_loss = sklearn.metrics.mean_squared_error(train_targets.cpu().numpy(), train_outputs_NTK_no_bias) # .numpy() only takes tensor in CPU

plot_learned_function(test_inputs_augmented[:,0],test_inputs_augmented[:,1],test_targets_augmented,0,'ground_truth')
plot_learned_function(test_inputs_augmented[:,0],test_inputs_augmented[:,1],test_outputs_NTK,NTK_loss[-1],'test_NTK')
plot_learned_function(test_inputs[:,0],test_inputs[:,1],test_outputs_NTK_no_bias,NTK_no_bias_loss,'test_NTK_no_bias')

plot_learned_function(train_inputs_augmented[:,0],train_inputs_augmented[:,1],train_outputs_NTK,NTK_train_loss,'train_NTK')
plot_learned_function(train_inputs[:,0],train_inputs[:,1],train_outputs_NTK_no_bias,NTK_no_bias_train_loss,'train_NTK_no_bias')

print('🧪')