# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import numpy
import torch
import matplotlib.pyplot
import os
import math
import utils

class NeuralNetwork(torch.nn.Module):
    def __init__(self, N, m):
        super(NeuralNetwork, self).__init__()
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
        output = output*math.sqrt(2/self.output_layer.in_features)

        return output

class NeuralNetworkASI(torch.nn.Module):
    def __init__(self, N, m):
        super(NeuralNetworkASI, self).__init__()
        
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
        output1 = output1*math.sqrt(2/self.output_layer1.in_features)
 
        output2 = self.hidden_layer2(input)
        output2 = torch.nn.functional.relu(output2)
        output2 = self.output_layer2(output2)
        output2 = output2*math.sqrt(2/self.output_layer2.in_features)

        output = (math.sqrt(2)/2)*(output1 - output2)

        return output

def get_loss(model, dataset):
    was_in_training=model.training
    model.train(False)

    (inputs, targets) = dataset[:]
    outputs = model(inputs) 
    outputs = outputs.reshape(-1) # Otherwise, outputs.shape=torch.Size([n, 1]) while loss_object() expects two tensors of the same shape and type
    loss_object = torch.nn.MSELoss()
    loss = loss_object(outputs, targets)

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

def phi(nn, input):
    # Forward
    output = nn(input)

    # Backward
    nn.zero_grad()
    output.backward()
    
    temp = torch.tensor([])

    for param in nn.parameters():
        temp=torch.cat((temp, param.grad.reshape(-1)))

    return temp

def Kw(nn,a,b):
    return torch.dot(phi(nn,a),phi(nn,b))

def Kw_matrix(inputs,nn):

    out = torch.empty((inputs.shape[0],inputs.shape[0]))

    for i, a in enumerate(inputs):
        for j, b in enumerate(inputs):
            out[i][j] = Kw(nn,a,b)
    
    return out

def has_converged(a, alpha, beta):
    y1 = math.log(a[int(len(a)/alpha)],10)
    y2 = math.log(a[-1],10)
    dy = y2-y1
    range_y = math.log(max(a),10)-math.log(min(a),10)
    if range_y == 0: slope=0
    else: slope = dy/(range_y*(1-1/alpha))
    
    print(f'slope={slope}')
    if slope > beta:
        return True

    return False

def train(nn, optimizer, train_dataset, batch_size, alpha, beta, path):
    was_in_training = nn.training
    nn.train(True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    (train_inputs, train_targets) = train_dataset[:]

    epoch_values = [] # We do not know when the training will finish
    train_loss_values = [] # train_loss_values[i]=The train loss in the BEGINNING of the i-th epoch

    epoch = 0
    while(True):
        
        epoch_values.append(epoch)
        train_loss_values.append(get_loss(nn, train_dataset))

        if (epoch+1)%alpha==0:
            print(f'm={nn.output_layer1.in_features}, epoch={epoch}, train_loss={train_loss_values[-1]}, ',end="")
            if has_converged(train_loss_values, alpha, beta):
                break

	    # If we do not have convergence...
        for batch, (inputs, targets) in enumerate(train_loader):
            
            # Forward
            outputs = nn(inputs)
            outputs = outputs.reshape(-1)
            loss_object = torch.nn.MSELoss()
            loss = loss_object(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch += 1

    # Create training curves
    fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'm={nn.output_layer1.in_features}')

    axs.plot(epoch_values, train_loss_values, linestyle='-', marker='o', color=utils.BLUE)
    axs.set_xlabel('epoch')
    axs.set_ylabel('train_loss')
    axs.grid()
    axs.set_yscale('log')

    os.makedirs(path, exist_ok=True)
    fig.savefig(path + f"/m={nn.output_layer1.in_features}.pdf")
    matplotlib.pyplot.close(fig)

    nn.train(was_in_training)