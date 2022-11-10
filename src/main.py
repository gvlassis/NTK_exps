# SPDX-FileCopyrightText: © 2022 Project's authors 
# SPDX-License-Identifier: MIT

import numpy
import torch
import matplotlib.pyplot
import time
import cpuinfo
import os
import sklearn.svm
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.metrics
import math
import shutil

# Hyperparameters
n_TEST = 200 
NUM_EXP_NN = 1 # The number of experiments for each (N,n_train) tuple. In each experiment we use a new training dataset but the same testing dataset (V remains the same)
NUM_EXP_NTK = 1
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MAX_WIDTH = 2000
STEP_WIDTH = 500

# Hyperparameters
BATCH_SIZE = 1
LR = 0.01
MOMENTUM = 0.0

# Stopping criteria 
ALPHA = 4
BETA = -0.2

class ManualDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.len = len(X)
        
        self.X = torch.tensor(X).reshape((self.len,1)).float()
        self.Y = torch.tensor(Y).float()

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class NeuralNetwork(torch.nn.Module):
    def __init__(self, m):
        super(NeuralNetwork, self).__init__()
        self.m = m
        self.hidden_layer = torch.nn.Linear(1, m, bias=False)
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
        
        self.hidden_layer1 = torch.nn.Linear(1, m, bias=False)
        self.output_layer1 = torch.nn.Linear(m, 1, bias=False)
        self.hidden_layer1.weight.data.normal_(mean=0.0, std=1)
        self.output_layer1.weight.data.normal_(mean=0.0, std=1)

        self.hidden_layer2 = torch.nn.Linear(1, m, bias=False)
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

def print_1D(axs, inputs, outputs):
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.grid()

    for i, input in enumerate(inputs):
        axs.plot(input[0], outputs[i], linestyle='None', marker='o', color='#039be5')
  
def get_loss_and_visualize_1D(model, dataset, m, exp):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_function() expects two tensors of the same shape and type
    loss = loss_function(outputs, targets)
    
    fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'm={m}, exp={exp}')

    print_1D(axs, inputs.numpy(), outputs.detach().numpy())

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'visualization/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

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

def visualize_NTK(inputs, outputs, exp):
    fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'NTK')

    print_1D(axs, inputs, outputs)

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'visualization/NTK/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

def train(model, train_loader, optimizer, train_dataset, m, exp):
    was_in_training = model.training
    model.train(True)

    epoch_values = [] # We do not know when the training will finish
    train_loss_values = [] # train_loss_values[i]=The train loss in the BEGINNING of the i-th epoch

    epoch = 0
    while(True):
        
        epoch_values.append(epoch)
        train_loss_values.append(get_loss(model, train_dataset))

        if (epoch+1)%ALPHA==0:
            print(f'm={m}, exp={exp}, epoch={epoch}, train_loss={train_loss_values[-1]}, ',end="")
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
    fig, axs = matplotlib.pyplot.subplots(ncols=2, figsize=[20, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'm={m}, exp={exp}')

    axs[0].plot(epoch_values, train_loss_values, linestyle='-', marker='o', color='#039be5')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('train_loss')
    axs[0].grid()
    axs[0].set_yscale('log')

    model.train(False)
    (inputs, _) = train_dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1])
    print_1D(axs[1], inputs.numpy(), outputs.detach().numpy())

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

    model.train(was_in_training)

    return get_loss_and_visualize_1D(model, test_dataset, m, exp)

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, 'training_curves/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)
dir_to_clean = os.path.join(script_dir, 'main_plots/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)
dir_to_clean = os.path.join(script_dir, 'visualization/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)

# Get CPU (and GPU) info for logging purposes 
cpu = cpuinfo.get_cpu_info()['brand_raw'] + '[x' + str(cpuinfo.get_cpu_info()['count']) + ']'
if DEVICE_TYPE == 'cuda':
    device_name = torch.cuda.get_device_name(DEVICE)
else:
    device_name = ''

start_time = time.time()

test_dataset = ManualDataset(numpy.linspace(-1, 1, n_TEST), numpy.zeros(n_TEST))

fig, axs = matplotlib.pyplot.subplots(figsize=[5, 5], dpi=100, tight_layout=True) # 500x500 plot

m_values = range(1, MAX_WIDTH, STEP_WIDTH)
loss_nn_values = numpy.empty([len(m_values), NUM_EXP_NN])
for j, m in enumerate(m_values):
    for exp in range(NUM_EXP_NN):
        train_dataset = ManualDataset([-0.6, 0.2, 0.8], [2, 0.5, 1])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        nn = NeuralNetworkASI(m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)

        # Run the experiment
        loss_nn_values[j, exp] = train(nn, train_loader, optimizer, train_dataset, m, exp)
    
loss_nn_values_mean = numpy.mean(loss_nn_values, axis=1)
loss_nn_values_std = numpy.std(loss_nn_values, axis=1)
axs.errorbar(x=m_values, y=loss_nn_values_mean, yerr=loss_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
axs.set_xlabel('m')
axs.set_ylabel('loss')
axs.grid()

(test_inputs, test_targets) = test_dataset[:]
test_inputs = test_inputs.cpu().numpy() # .numpy() only takes tensor in CPU
test_targets = test_targets.cpu().numpy() # .numpy() only takes tensor in CPU

# Train the NTKs
loss_NTK_values = numpy.empty([NUM_EXP_NTK])
for exp in range(NUM_EXP_NTK):
    train_dataset = ManualDataset([-0.6, 0.2, 0.8], [2, 0.5, 1])
    (train_inputs, train_targets) = train_dataset[:]
    train_inputs = train_inputs.cpu().numpy() # .numpy() only takes tensor in CPU
    train_targets = train_targets.cpu().numpy() # .numpy() only takes tensor in CPU

    NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
    print(f'Training NTK#{exp}')
    NTK.fit(train_inputs, train_targets)
    print(f'Infering NTK#{exp}')
    test_outputs = NTK.predict(test_inputs)
    loss_NTK_values[exp] = sklearn.metrics.mean_squared_error(test_targets, test_outputs)
    visualize_NTK(test_inputs, test_outputs, exp)

loss_NTK_values_mean = numpy.mean(loss_NTK_values)
loss_NTK_values_std = numpy.std(loss_NTK_values)

axs.plot(m_values, (loss_NTK_values_mean+loss_NTK_values_std) * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#8e0000')
axs.plot(m_values, loss_NTK_values_mean * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')
axs.plot(m_values, (loss_NTK_values_mean-loss_NTK_values_std) * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#8e0000')

end_time = time.time()
fig.suptitle(f'Time: {end_time-start_time:.0f}s, CPU: {cpu}, DEVICE_TYPE={DEVICE}[{device_name}]\n'
             f'n_TEST={n_TEST}, NUM_EXP_NN={NUM_EXP_NN}, NUM_EXP_NTK={NUM_EXP_NTK}\n'
             f'ALPHA={ALPHA}, BETA={BETA}\n')

script_dir = os.path.dirname(__file__)
fig_dir = os.path.join(script_dir, 'main_plots/')
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

fig.savefig(fig_dir + f'N=1.pdf')
matplotlib.pyplot.close(fig)

print('Done!')

