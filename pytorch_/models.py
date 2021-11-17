from .src.networks import *
from .src.base import *
from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

class f3dasm_model():

    def __init__(self, network, loss, optimizer, epochs=10): 

        """ Initialize """

        self.epochs = epochs
        self.net = network
        self.loss_func = loss
        self.optimizer = optimizer

    def __call__(self, dataset):

        """ Call with a dataset """

        self.dataset = dataset

    def train(self):

        """ Method: Training """

        return train(self.dataset.train_loader, self.net,
                                self.optimizer, self.loss_func).item()
    def test(self):

        """ Method: Testing """

        return test(self.dataset.test_loader, self.net,
                                self.loss_func).item()
    def run(self):

        """ Method: Run Model with desired number of epochs"""

        self.train_loss = [ ]
        self.test_loss = [ ]
        self.epoch = [ ]
        for count in tqdm(range(self.epochs)):

            count += 1
            self.train_loss.append(self.train())
            self.test_loss.append(self.test())
            self.epoch.append(count)

        losses = np.array([self.epoch, self.train_loss, self.test_loss]).T
        self.losses = pd.DataFrame(losses, columns= ['epoch','train','test'])
    
    def save(self,filename):

        """ Method: Run Model with desired number of epochs"""

        name, ext = os.path.splitext(filename)
        assert ext == '.pt' or ext == '.pth'
        torch.save(self.net,filename)

class HyperParamSearch_xval():
    
    def __init__(self, dataset, network, loss_func, optimizer):

        self.dataset = dataset
        self.network = network
        self.loss_func = loss_func
        self.optimizer = optimizer

    def __call__(self, neuron=10, hidden=1, lr=0.001, epochs=1000, activation_tag='sigmoid', batch_size=32):

        self.epochs = epochs
        self.batch_size = batch_size

        self.network = self.network(in_feature=self.dataset.in_size, n_neuron=neuron, out_feature=self.dataset.out_size, n_hidden=hidden, activation_tag=activation_tag)
        self.loss_func = self.loss_func()
        self.optimizer = self.optimizer(self.network.parameters(),lr=lr)

    def run(self):

        train, test = xval(self.dataset, self.network, self.optimizer, self.loss_func,  epochs=self.epochs)
        
        return train[0], test[0], train[1], test[1]

    def flush(self,filename):
        name, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            self.losses.to_pickle(filename)

        elif ext == ".csv":
            self.losses.to_csv(filename)
        else:
            NotImplementedError 

    def save(self,filename,state=False):
        name, ext = os.path.splitext(filename)
        assert ext == '.pt' or ext == '.pth'
        if state:
            torch.save(self.network.state_dict(),filename)
        else:
            torch.save(self.network,filename)



##########################################################################################
# PURGATORY
##########################################################################################
    
class Bessa2017():
    
    def __init__(self, dataset):

        self.dataset = dataset

    def __call__(self, n_hidden=10, epochs=50):

        self.epochs = epochs
        
        self.net = SingleNet(self.dataset.in_size,n_hidden,self.dataset.out_size)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001) 
        #self.optimizer = torch.optim.SGD(self.net.parameters(), lr=5e-1)

        self.train_loss = []
        self.test_loss = []
        self.epoch = []

    def train_test(self):

        for count in tqdm(range(self.epochs)):

            count += 1
            self.train_loss.append(train(self.dataset.train_loader, self.net,
                                    self.optimizer, self.loss_func).item())
            self.test_loss.append(test(self.dataset.test_loader, self.net,
                                    self.loss_func).item())
            self.epoch.append(count)

        losses = np.array([self.epoch, self.train_loss, self.test_loss]).T
        self.losses = pd.DataFrame(losses, columns= ['epoch','train','test'])
    
    def flush(self,filename):
        name, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            self.losses.to_pickle(filename)

        elif ext == ".csv":
            self.losses.to_csv(filename)
        else:
            NotImplementedError 

    def save(self,filename):
        name, ext = os.path.splitext(filename)
        assert ext == '.pt' or ext == '.pth'
        torch.save(self.net,filename)

        

        








