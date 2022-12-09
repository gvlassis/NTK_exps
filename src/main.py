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

# Parameters
n_TEST = 200
NUM_EXP = 20
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MAX_WIDTH_EXPON = 13

# Hyperparameters
BATCH_SIZE = 1
LR = 0.01
MOMENTUM = 0.0

# Stopping criteria 
ALPHA = 4
BETA = -0.2

# Colors
RED = '#d32f2f'
GREEN = '#7cb342'
BLUE = '#039be5'

class ManualDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.len = len(X)
        
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return self.len

class NeuralNetwork(torch.nn.Module):
    def __init__(self, m):
        super(NeuralNetwork, self).__init__()
        self.m = m
        self.hidden_layer = torch.nn.Linear(2, m, bias=False)
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
        
        self.hidden_layer1 = torch.nn.Linear(2, m, bias=False)
        self.output_layer1 = torch.nn.Linear(m, 1, bias=False)
        self.hidden_layer1.weight.data.normal_(mean=0.0, std=1)
        self.output_layer1.weight.data.normal_(mean=0.0, std=1)

        self.hidden_layer2 = torch.nn.Linear(2, m, bias=False)
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

class PiecewiseLinearFunction:
    def __init__(self, xs, ys):
        self.changes = len(xs)-1

        self.xs = numpy.array(xs)
        inds = self.xs.argsort()
        self.xs = self.xs[inds]

        self.ys = numpy.array(ys)[inds]

        self.ws = numpy.empty((self.changes,2))
        for i in range(0,self.changes):
            p1 = [self.xs[i],self.ys[i]]
            p2 = [self.xs[i+1],self.ys[i+1]]
            a,b = get_linear_weights(p1,p2)
            self.ws[i] = [a,b]
    
    def get_output(self, input):
        if input<self.xs[0]:
            output = self.ws[0,0]*input+self.ws[0,1]
            return output
        for i in range(0,self.changes): 
            if input<self.xs[i+1]:
                output = self.ws[i,0]*input+self.ws[i,1]
                return output
        #input>self.xs[-1]=self.xs[changes]
        output = self.ws[-1,0]*input+self.ws[-1,1]
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

def train(model, optimizer, train_dataset, m, exp, test_dataset, test_outputs_NTK):
    was_in_training = model.training
    model.train(True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    (train_inputs, train_targets) = train_dataset[:]
    train_inputs = train_inputs.cpu().numpy() # .numpy() only takes tensor in CPU
    train_targets = train_targets.cpu().numpy() # .numpy() only takes tensor in CPU

    (test_inputs,_) = test_dataset[:]

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

    axs[0].plot(epoch_values, train_loss_values, linestyle='-', marker='o', color=BLUE)
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('train_loss')
    axs[0].grid()
    axs[0].set_yscale('log')

    axs[1].set_xlabel('theta')
    axs[1].set_ylabel('y')
    axs[1].grid()
    model.train(False)
    test_outputs = model(test_inputs) 
    test_outputs = test_outputs.reshape(-1) # Otherwise, test_outputs.shape=torch.Size([n, 1])
    print_1D(axs[1], test_inputs, test_outputs.cpu().detach().numpy(), BLUE)
    print_1D(axs[1], test_inputs, test_outputs_NTK, GREEN)
    print_1D(axs[1], train_inputs, train_targets, RED)

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

    model.train(was_in_training)

def print_1D(axs, inputs, outputs, color):
    for i, input in enumerate(inputs):
        (_, theta) = get_polar_from_cart(input[0], input[1])
        axs.plot(theta, outputs[i], linestyle='None', marker='o', color=color)

def get_polar_from_cart(x,y):
    r = math.sqrt(x**2+y**2)

    if x==0:
        if y>0:
            theta = math.pi/2
        elif y<0:
            theta = (3/2)*math.pi
        else:
            theta = None 
    else:
        if x>0 and y>=0:
            theta = math.atan(y/x)
        elif x<0 and y>=0:
            theta = math.atan(y/x)+math.pi
        elif x<0 and y<0:
            theta = math.atan(y/x)+math.pi
        else:
            theta = math.atan(y/x)+2*math.pi
    
    return r, theta

def get_points_from_thetas(thetas): 
    xs = 1*numpy.cos(thetas)
    xs = xs.reshape((-1,1))
    ys = 1*numpy.sin(thetas)
    ys = ys.reshape((-1,1))

    points = numpy.concatenate((xs,ys),axis=1)

    return points 

def get_linear_weights(p1,p2):
    (x1,y1) = p1
    (x2,y2) = p2

    a = (y1-y2)/(x1-x2)
    b = y1 - a*x1
 
    return a,b

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, 'training_curves/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)
 
if DEVICE_TYPE == 'cuda':
    device_name = torch.cuda.get_device_name(DEVICE)
else:
    device_name = ''

train_thetas = [0, math.pi, math.pi*2/3]
train_inputs = get_points_from_thetas(train_thetas)
train_targets = [-1, 2, 1/2]
train_dataset = ManualDataset(train_inputs, train_targets)

test_thetas = numpy.linspace(0, 4*math.pi, n_TEST)
test_inputs = get_points_from_thetas(test_thetas)
f = PiecewiseLinearFunction(train_thetas,train_targets)
test_targets = numpy.vectorize(f.get_output)(test_thetas)
test_dataset = ManualDataset(test_inputs, test_targets)

# Train the NTK
NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
print(f'Training NTK')
NTK.fit(train_inputs, train_targets)
print(f'Infering NTK')
test_outputs_NTK = NTK.predict(test_inputs)
NTK_loss = sklearn.metrics.mean_squared_error(test_targets, test_outputs_NTK)

# Train the nns
m_exponents = range(1, MAX_WIDTH_EXPON+1)
m_values = [2**exp for exp in m_exponents]
nn_loss = numpy.empty([len(m_values), NUM_EXP])
for i, m in enumerate(m_values):
    for exp in range(NUM_EXP):
        nn = NeuralNetworkASI(m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)

        # Train the nn
        train(nn, optimizer, train_dataset, m, exp, test_dataset, test_outputs_NTK)

        nn_loss[i,exp] = get_loss(nn, test_dataset)

nn_loss_mean = numpy.mean(nn_loss, axis=1)
nn_loss_std = numpy.std(nn_loss, axis=1)

fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
axs.set_xlabel('m')
axs.set_ylabel('l2_loss')
axs.grid()
axs.set_xscale('log', base=2)

axs.errorbar(m_values, nn_loss_mean, nn_loss_std, linestyle='-', marker='o', color=BLUE, ecolor=RED, capsize=5)
axs.plot(m_values, NTK_loss*numpy.ones(len(m_values)), linestyle='-', marker='o', color=GREEN)

fig.savefig('l2_loss.pdf')
matplotlib.pyplot.close(fig)

print('Done!')