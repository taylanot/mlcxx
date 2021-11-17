import torch
from torch.autograd import Variable 
import torch.nn.functional as F
from collections import OrderedDict
from .util import network_tools, RBF
from .base import network_modules

class SingleNet(torch.nn.Module):
    def __init__(self, in_feature, n_hidden, out_feature):
        super(SingleNet,self).__init__()

        self.layer1 = torch.nn.Linear(in_feature, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, out_feature)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer2(self.sigmoid(self.layer1(x)))
        return x

class SimpleNet(network_tools,torch.nn.Module):
    def __init__(self, in_feature, n_hidden, out_feature, activation_tag='softsign'):
        super(SimpleNet, self).__init__()

        self.layer1 = torch.nn.Linear(in_feature, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, out_feature)
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU}
        self.activation = self.activations[activation_tag]()

    def forward(self,x):
        x = self.layer2(self.activation(self.layer1(x)))
        return x

class DeepNet(network_tools,torch.nn.Module):
    def __init__(self, in_feature, n_neuron, out_feature, n_hidden=0, activation_tag='softsign'):
        super(DeepNet, self).__init__()

        self.layer_in = torch.nn.Linear(in_feature, n_neuron,bias=True)
        self.layer_out = torch.nn.Linear(n_neuron, out_feature,bias=True)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_neuron, n_neuron, bias=True) for i in range(n_hidden)])
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU, 'tanshrink':torch.nn.Tanhshrink}
        self.activation = self.activations[activation_tag]()

    def forward(self,x):

        x = self.activation(self.layer_in(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        
        x = self.layer_out(x)

        return x

class LinearNetwork(network_modules, network_tools, torch.nn.Module):
    def __init__(self, in_feature, out_feature,  bias=True):
        super(LinearNetwork, self).__init__()
        self.bias = bias
        self.layer = torch.nn.Linear(in_feature, out_feature, bias=bias)
    def forward(self,x, weights=None):
        if weights is None:
            return self.layer(x)
        else:
            if self.bias:
                return F.linear(x, weights[0], weights[1])
            else:
                return F.linear(x, weights[0])

class LinearReg(network_modules, network_tools, torch.nn.Module):
    def __init__(self,basis_functions=[lambda x: torch.ones_like(x), lambda x: x, lambda x: x**2, torch.sin, torch.cos]):
        super(LinearReg, self).__init__()

        self.basis_functions = basis_functions
        self.num= len(basis_functions) 
        self.weights = torch.nn.Linear(self.num,1,bias=False)
        self.loss_func = torch.nn.MSELoss()

    def features(self,x):
        feat = [self.basis_functions[i](x) for i in range(self.num)]
        return torch.cat(feat,1)

    def analytic(self, D):
        x,y = D
        X = self.features(x)
        weights = (X.T@X).inverse()@X.T@y
        return weights.reshape(1,-1)

    def update_weights(self, weights):
        with torch.no_grad():
            self.weights.weight = torch.nn.Parameter(weights)
            self.parameter=torch.nn.Parameter(weights)

    def forward(self,x, weights=None):
        x = self.features(x)
        if weights is None:
            return self.weights(x)
        else:
            return  x @ weights[0].reshape(-1,1)   

    def analytic_fit(self, D):
        self.update_weights(self.analytic(D))

    def gd_fit(self,D, epochs=10):
        x,y = D
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.1)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.forward(x); loss = self.loss_func(pred,y)
            loss.backward()
            #self.optimizer.step(lambda: self.loss_func(pred,y))
            self.optimizer.step()
        #print(self.weights.weight)

    def reset(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

class RBFNetwork(network_modules, network_tools, torch.nn.Module):
    
    def __init__(self, in_features=1, n_neurons=100, out_features=1):
        super(RBFNetwork, self).__init__()
        self.features = RBF(1, n_neurons, functional_key='gaussian')
        self.linear = (torch.nn.Linear(n_neurons, out_features))
    
    def forward(self, x, weights=None):
        x = self.features(x)
        if weights is None:
            return self.linear(x)
        else:
            return F.linear(x, weights[0], weights[1])

    def gd_fit(self,D, epochs=10):
        x,y = D
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
        self.loss_func = torch.nn.MSELoss()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.forward(x); loss = self.loss_func(pred,y)
            loss.backward()
            #self.optimizer.step(lambda: self.loss_func(pred,y))
            self.optimizer.step()
        #print(list(self.parameters()))

    def analytic(self, D):
        x,y = D
        X = self.features(x)
        bias = torch.ones_like(x)
        X = torch.cat([bias,X],1)
        if X.shape[0] > X.shape[1]:
            weights = (torch.pinverse(X.T@X))@X.T@y
        else:
        #weights = (torch.tensor(np.linalg.pinv(X.T@X)))@X.T@y
            weights = torch.lstsq(y, X).solution
        return (weights.reshape(1,-1))

    def update_weights(self, weights):
        self.linear.bias = torch.nn.Parameter(torch.unsqueeze(weights[0,0],0))
        self.linear.weight = torch.nn.Parameter(torch.unsqueeze(weights[0,1:],0))

    def analytic_fit(self, D):
        self.update_weights(self.analytic(D))
        #print(list(self.parameters()))



class SineNetwork(network_modules, network_tools, torch.nn.Module):
    def __init__(self, in_feature, out_feature, n_neurons=40):
        super(SineNetwork, self).__init__()
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.Sequential(OrderedDict([
            ('l1',torch.nn.Linear(in_feature,n_neurons)),
            ('l2',torch.nn.Linear(n_neurons,n_neurons)),
            ('l3',torch.nn.Linear(n_neurons,out_feature))]))

    def forward(self,x, weights=None):

        if weights is None:
            for no, layer in enumerate(self.layers):
                x = layer(x) 
                if no != len(self.layers)-1:
                    x = self.activation(x)
            return x
        else:
            return self.argforward(x,weights)

    def argforward(self,x, weights=None):
        for no, layer in enumerate(self.layers):
            idx = 2*no
            if type(layer) is torch.nn.Linear:
                x = torch.nn.functional.linear(x, weights[idx], weights[idx+1])
                if no != len(self.layers)-1:
                    x = self.activation(x)
        return x


class FeedForward(network_modules, network_tools,torch.nn.Module):
    def __init__(self, in_feature, out_feature, **kwargs):
        super(FeedForward, self).__init__()
        
        activation_tag = kwargs['architecture']['activation_tag']
        n_neuron = kwargs['architecture']['neuron_hidden'][0]
        n_hidden = kwargs['architecture']['neuron_hidden'][1]

        self.layer_in = torch.nn.Linear(in_feature, n_neuron,bias=True)
        self.layer_out = torch.nn.Linear(n_neuron, out_feature,bias=True)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_neuron, n_neuron, bias=True) for i in range(n_hidden)])
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU, 'tanshrink':torch.nn.Tanhshrink}
        self.activation = self.activations[activation_tag]()

        self.losses = {'MSELoss': torch.nn.MSELoss}
        self.optimizers = {'Adam': torch.optim.Adam}

        self.loss_func = self.losses[kwargs['training']['loss_func']]()
        self.optimizer = self.optimizers[kwargs['optimizer']['algo']](self.parameters(),lr=kwargs['optimizer']['lr'])

    def forward(self,x):

        x = self.activation(self.layer_in(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        
        x = self.layer_out(x)

        return x




        
        



        
