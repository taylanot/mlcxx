from .src.networks import *
from .src.base import *
from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

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
    
