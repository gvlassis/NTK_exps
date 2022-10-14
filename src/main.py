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
N = 10
n_TRAIN = 10000
n_TEST = 10000
NUM_EXP = 5 # The number of experiments for each (N,n_train) tuple. In each experiment we use a new training dataset but the same testing dataset (V remains the same)
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MAX_TIMES_N = 60
STEP_TIMES_N = 3

# Hyperparameters
BATCH_SIZE = 25
LR = 0.01
MOMENTUM = 0.99

# Stopping criteria 
ALPHA = 4
BETA = -0.1

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
    range_y=math.log(max(a),10)-math.log(min(a),10)
    slope = dy/(range_y*(1-1/ALPHA))
    print(f'slope={slope}')
    if slope > BETA:
        return True

    return False

def train(model, train_loader, optimizer, train_dataset, N, m, exp):
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
    fig, axs = matplotlib.pyplot.subplots(figsize=[10, 10], dpi=100, tight_layout=True)
    fig.suptitle(f'm={m}, exp={exp}')

    axs.plot(epoch_values, train_loss_values, linestyle='-', marker='o', color='#039be5')
    axs.set_xlabel('epoch')
    axs.set_ylabel('train_loss')
    axs.grid()
    axs.set_yscale('log')

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'training_curves/m={0}/'.format(m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

    model.train(was_in_training)

    return get_loss(model, test_dataset)

# Clean working directory
script_dir = os.path.dirname(__file__)
dir_to_clean = os.path.join(script_dir, 'training_curves/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)
dir_to_clean = os.path.join(script_dir, 'main_plots/')
if os.path.isdir(dir_to_clean): shutil.rmtree(dir_to_clean)

# Get CPU (and GPU) info for logging purposes 
cpu = cpuinfo.get_cpu_info()['brand_raw'] + '[x' + str(cpuinfo.get_cpu_info()['count']) + ']'
if DEVICE_TYPE == 'cuda':
    device_name = torch.cuda.get_device_name(DEVICE)
else:
    device_name = ''

distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N,device=DEVICE), torch.eye(N,device=DEVICE)) # device=DEVICE is essential to get RNG in the GPU 

start_time = time.time()

test_dataset = SphereDataset(distribution, n_TEST)

fig, axs = matplotlib.pyplot.subplots(figsize=[5, 5], dpi=100, tight_layout=True) # 500x500 plot

m_values = range(N, MAX_TIMES_N*N, STEP_TIMES_N*N)
loss_nn_values = numpy.empty([len(m_values), NUM_EXP])
for j, m in enumerate(m_values):
    for exp in range(NUM_EXP):
        train_dataset = SphereDataset(distribution, n_TRAIN, test_dataset.V)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        nn = NeuralNetwork(N,m)
        nn.to(DEVICE)

        # Set up the optimizer for the nn
        optimizer = torch.optim.SGD(nn.parameters(), lr=LR, momentum=MOMENTUM)

        # Run the experiment
        loss_nn_values[j, exp] = train(nn, train_loader, optimizer, train_dataset, N, m, exp)

loss_nn_values_mean = numpy.mean(loss_nn_values, axis=1)
loss_nn_values_std = numpy.std(loss_nn_values, axis=1)
axs.errorbar(x=m_values, y=loss_nn_values_mean, yerr=loss_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
axs.set_xlabel('m')
axs.set_ylabel('loss')
axs.set_title(f'n_TRAIN={n_TRAIN}')
axs.grid()

(train_inputs, train_targets) = train_dataset[:]
train_inputs = train_inputs.cpu().numpy() # .numpy() only takes tensor in CPU
train_targets = train_targets.cpu().numpy() # .numpy() only takes tensor in CPU

(test_inputs, test_targets) = test_dataset[:]
test_inputs = test_inputs.cpu().numpy() # .numpy() only takes tensor in CPU
test_targets = test_targets.cpu().numpy() # .numpy() only takes tensor in CPU

# Train the NTK(with the last train_dataset)
NTK = sklearn.kernel_ridge.KernelRidge(alpha=1e-10, kernel=K)
print('Training the NTK')
NTK.fit(train_inputs, train_targets)
print('Infering with the NTK')
loss_NTK = sklearn.metrics.mean_squared_error(test_targets,NTK.predict(test_inputs))

axs.plot(m_values, loss_NTK * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')

end_time = time.time()
fig.suptitle(f'N={N} (Time: {end_time-start_time:.0f}s, CPU: {cpu}, DEVICE_TYPE={DEVICE}[{device_name}])\n'
             f'n_TEST={n_TEST}, NUM_EXP={NUM_EXP}\n'
             f'TRAIN_LOSS1={TRAIN_LOSS1}, EPOCHS2={EPOCHS2}, TOTAL_BATCHES3={TOTAL_BATCHES3}, TRAIN_LOSS_DIF4={TRAIN_LOSS_DIF4}, EPOCH_DIF4={EPOCH_DIF4}\n')

script_dir = os.path.dirname(__file__)
fig_dir = os.path.join(script_dir, 'main_plots/')
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

fig.savefig(fig_dir + f'N={N}.pdf')
matplotlib.pyplot.close(fig)

print('Done!')

