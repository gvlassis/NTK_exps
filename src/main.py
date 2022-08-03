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
N_VALUES = [10, 50, 100]
n_TRAIN_VALUES = [1000, 2500, 5000, 7500, 10000]
n_TEST = 10000
NUM_EXP = 20 # The number of experiments for each (N,n_train) tuple. In each experiment we use a new training dataset but the same testing dataset (V remains the same)
DEVICE_TYPE = 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
loss_function = torch.nn.MSELoss()
MAX_TIMES_N = 10
STEP_TIMES_N = 1

# Hyperparameters
BATCH_SIZE = 25

TRAIN_LOSS1 = 1e-1
EPOCHS2 = 20
TOTAL_BATCHES3_MIN = 5000
BATCHES_PER_EPOCH = [ math.ceil(n_train/BATCH_SIZE) for n_train in n_TRAIN_VALUES]
LCM = numpy.lcm.reduce(BATCHES_PER_EPOCH)
COEFF = math.ceil(TOTAL_BATCHES3_MIN/LCM) # Minimum s.t: LCM*COEFF>=TOTAL_BATCHES3_MIN
TOTAL_BATCHES3 = LCM*COEFF 
TRAIN_LOSS_DIF4 = 0.001
EPOCH_DIF4 = 100

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

def train(model, train_loader, optimizer, train_dataset, N, n_train, m, exp):
    was_in_training = model.training
    model.train(True)

    # Flags for the stopping criteria 
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False

    batches_per_epoch = math.ceil(n_train/BATCH_SIZE)

    epoch_values = [] # We do not know when the training will finish
    train_loss_values = [] # train_loss_values[i]=The train loss in the BEGINNING of the i-th epoch

    epoch = 0
    while (True):
        
        epoch_values.append(epoch)
        train_loss_values.append(get_loss(model, train_dataset))

        if not(flag1) and train_loss_values[epoch] < TRAIN_LOSS1:
            test_loss1 = get_loss(model, test_dataset)
            flag1 = True
            if(flag2 and flag3 and flag4):
                print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}, flag1={flag1}, flag2={flag2}, flag3={flag3}, flag4={flag4}')
                break

        if not(flag2) and epoch == EPOCHS2:
            test_loss2 = get_loss(model, test_dataset)
            flag2 = True 
            if(flag1 and flag3 and flag4):
                print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}, flag1={flag1}, flag2={flag2}, flag3={flag3}, flag4={flag4}')
                break

        if not(flag3) and epoch*batches_per_epoch == TOTAL_BATCHES3:
            test_loss3 = get_loss(model, test_dataset)
            flag3 = True
            if(flag1 and flag2 and flag4):
                print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}, flag1={flag1}, flag2={flag2}, flag3={flag3}, flag4={flag4}')
                break

        # The 4th stopping criterion can only be triggered if we have trained for at least EPOCH_DIF4 epochs
        if not(flag4) and epoch+1>=EPOCH_DIF4 and ((train_loss_values[epoch-EPOCH_DIF4]-train_loss_values[epoch])<TRAIN_LOSS_DIF4):
            test_loss4 = get_loss(model, test_dataset)
            flag4 = True
            if(flag1 and flag2 and flag3):
                print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}, flag1={flag1}, flag2={flag2}, flag3={flag3}, flag4={flag4}')
                break
        
        if epoch%10==0: print(f'N={N}, n_train={n_train}, m={m}, exp={exp}, epoch={epoch}, flag1={flag1}, flag2={flag2}, flag3={flag3}, flag4={flag4}')

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
    fig.suptitle(f'N={N}, n_train={n_train}, m={m}, exp={exp}')

    axs.plot(epoch_values, train_loss_values, linestyle='-', marker='o', color='#039be5')
    axs.set_xlabel('epoch')
    axs.set_ylabel('train_loss')
    axs.grid()
    axs.set_yscale('log')

    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.join(script_dir, 'training_curves/N={0}/n_train={1}/m={2}/'.format(N, n_train, m))
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + f'exp={exp}.pdf')
    matplotlib.pyplot.close(fig)

    model.train(was_in_training)

    return test_loss1, test_loss2, test_loss3, test_loss4

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

for N in N_VALUES:
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N,device=DEVICE), torch.eye(N,device=DEVICE)) # device=DEVICE is essential to get RNG in the GPU 
    
    start_time = time.time()

    test_dataset = SphereDataset(distribution, n_TEST) # We have one V for each N
    
    # One figure for each N
    fig, axs = matplotlib.pyplot.subplots(nrows=4, ncols=len(n_TRAIN_VALUES), figsize=[len(n_TRAIN_VALUES)*5, 4*5], dpi=100, tight_layout=True) # 500x500 plots

    m_values = range(N, MAX_TIMES_N*N, STEP_TIMES_N*N)
    for i, n_train in enumerate(n_TRAIN_VALUES):
        loss1_nn_values = numpy.empty([len(m_values), NUM_EXP])
        loss2_nn_values = numpy.empty([len(m_values), NUM_EXP])
        loss3_nn_values = numpy.empty([len(m_values), NUM_EXP])
        loss4_nn_values = numpy.empty([len(m_values), NUM_EXP])
        for j, m in enumerate(m_values):
            for exp in range(NUM_EXP):
                train_dataset = SphereDataset(distribution, n_train, test_dataset.V)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                nn = NeuralNetwork(N,m)
                nn.to(DEVICE)

                # Set up the optimizer for the nn
                optimizer = torch.optim.Adam(nn.parameters())

                # Run the experiment
                loss1_nn_values[j, exp], loss2_nn_values[j, exp], loss3_nn_values[j, exp], loss4_nn_values[j, exp] = train(nn, train_loader, optimizer, train_dataset, N, n_train, m, exp)

        # 4 subplots for each n
        loss1_nn_values_mean = numpy.mean(loss1_nn_values, axis=1)
        loss1_nn_values_std = numpy.std(loss1_nn_values, axis=1)
        axs[0, i].errorbar(x=m_values, y=loss1_nn_values_mean, yerr=loss1_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
        axs[0, i].set_xlabel('m')
        axs[0, i].set_ylabel('loss1')
        axs[0, i].set_title(f'n_train={n_train}')
        axs[0, i].grid()

        loss2_nn_values_mean = numpy.mean(loss2_nn_values, axis=1)
        loss2_nn_values_std = numpy.std(loss2_nn_values, axis=1)
        axs[1, i].errorbar(x=m_values, y=loss2_nn_values_mean, yerr=loss2_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
        axs[1, i].set_xlabel('m')
        axs[1, i].set_ylabel('loss2')
        axs[1, i].set_title(f'n_train={n_train}')
        axs[1, i].grid()

        loss3_nn_values_mean = numpy.mean(loss3_nn_values, axis=1)
        loss3_nn_values_std = numpy.std(loss3_nn_values, axis=1)
        axs[2, i].errorbar(x=m_values, y=loss3_nn_values_mean, yerr=loss3_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
        axs[2, i].set_xlabel('m')
        axs[2, i].set_ylabel('loss3')
        axs[2, i].set_title(f'n_train={n_train}')
        axs[2, i].grid()

        loss4_nn_values_mean = numpy.mean(loss4_nn_values, axis=1)
        loss4_nn_values_std = numpy.std(loss4_nn_values, axis=1)
        axs[3, i].errorbar(x=m_values, y=loss4_nn_values_mean, yerr=loss4_nn_values_std, linestyle='--', marker='o', color='#039be5', ecolor='#e53935', capsize=5)
        axs[3, i].set_xlabel('m')
        axs[3, i].set_ylabel('loss4')
        axs[3, i].grid()
        
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

        axs[0, i].plot(m_values, loss_NTK * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')
        axs[1, i].plot(m_values, loss_NTK * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')
        axs[2, i].plot(m_values, loss_NTK * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')
        axs[3, i].plot(m_values, loss_NTK * numpy.ones(len(m_values)), linestyle='-', marker=None, color='#fbc02d')

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

