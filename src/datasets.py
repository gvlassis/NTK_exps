# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import torch
import json

class HyperplaneDataset(torch.utils.data.Dataset):
    def __init__(self, V, X, Y):
        self.V = V
        self.X = X
        self.Y = Y
    
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.Y)

class DiscreteSphereDataset(HyperplaneDataset):
    def __init__(self, distribution, n, V):
        X = distribution.sample((n,))
        line_norms = torch.linalg.norm(X,dim=1)
        line_norms_T = torch.reshape(line_norms,(n,1))
        X = X/line_norms_T
  
        Y = torch.matmul(X,V)

        Y = Y >= 0 
        Y = Y.float()

        super().__init__(V,X,Y)

    @classmethod
    def create_without_V(cls, distribution, n):
        V = distribution.sample()
        V = V / torch.linalg.norm(V)
        
        return cls(distribution, n, V)

class ContinuousSphereDataset(HyperplaneDataset):
    def __init__(self, distribution, n, V):
        X = distribution.sample((n,))
        line_norms = torch.linalg.norm(X,dim=1)
        line_norms_T = torch.reshape(line_norms,(n,1))
        X = X/line_norms_T
  
        Y = torch.matmul(X,V)

        super().__init__(V,X,Y)

    @classmethod
    def create_without_V(cls, distribution, n):
        V = distribution.sample()
        V = V / torch.linalg.norm(V)
        
        return cls(distribution, n, V)

class UniformDataset(HyperplaneDataset):
    def __init__(self, distribution, n, V):
        X = distribution.sample((n,))
  
        Y = torch.matmul(X,V)

        Y = Y >= 0 
        Y = Y.float()

        super().__init__(V,X,Y)

    @classmethod
    def create_without_V(cls, distribution, n):
        V = distribution.sample()
        V = V / torch.linalg.norm(V)
        
        return cls(distribution, n, V)

class HyperplaneDatasetEncoder(json.JSONEncoder):
    def default(self, dataset):
            return {"V":dataset.V.tolist(), "X":dataset.X.tolist(), "Y":dataset.Y.tolist()}

def decode_hyperplane_dataset(dct):
    return HyperplaneDataset(torch.tensor(dct["V"]), torch.tensor(dct["X"]), torch.tensor(dct["Y"]))