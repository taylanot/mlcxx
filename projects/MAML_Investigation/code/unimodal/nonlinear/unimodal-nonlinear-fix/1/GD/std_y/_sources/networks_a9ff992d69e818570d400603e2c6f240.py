import torch
import torch.nn.functional as F
from collections import OrderedDict
class LinearNetwork(torch.nn.Module):
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

class NonlinearNetwork(torch.nn.Module):
    def __init__(self, in_feature, out_feature, n_hidden=1,  n_neuron=40,\
            activation_tag='tanh'):
        super(NonlinearNetwork, self).__init__()
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh,\
                'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU,\
                'tanshrink':torch.nn.Tanhshrink}
        self.activation = self.activations[activation_tag]()
        keys = ['hidden-'+str(i+1) for i in range(n_hidden)]
        vals = [torch.nn.Linear(n_neuron, n_neuron) for i in range(n_hidden)]
        fixed_layers = {'input':torch.nn.Linear(in_feature, n_neuron),
                        'output':torch.nn.Linear(n_neuron, out_feature)}
        layer_info = OrderedDict(zip(keys,vals))
        layer_info.update(fixed_layers)
        layer_info.move_to_end('input', last=False)

        self.layers = torch.nn.Sequential(layer_info)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self,x, weights=None):

        if weights is None:
            for no, layer in enumerate(self.layers):
                x = layer(x) 
                if no != len(self.layers)-1:
                    x = self.activation(x)
            return x
        else:
            return self.argforward(x,weights)

    def argforward(self,x, weights):
        for no, layer in enumerate(self.layers):
            idx = 2*no
            if type(layer) is torch.nn.Linear:
                x = torch.nn.functional.linear(x, weights[idx], weights[idx+1])
                if no != len(self.layers)-1:
                    x = self.activation(x)
        return x
    def fit(self, D, lr=0.001, n_iter=1, load=False):
        x, y = D
        if load:
            self.load()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_iter):
            optimizer.zero_grad()
            loss = self.loss_fn(self.forward(x), y)
            loss.backward()
            optimizer.step()

    def predict(self, x):
        return self.forward(x)

    @torch.no_grad()
    def test(self, D):
        x, y = D
        return self.loss_fn(self.forward(x), y)

    def load(self, path='model.pt'):
        self.load_state_dict(torch.load(path),strict=False)

#class MAML(torch.nn.Module):
#    def __init__(self, in_feature, n_neuron, out_feature, n_hidden=0, activation_tag='tanh'):
#        super(MAML, self).__init__()
#
#        self.layer_in = torch.nn.Linear(in_feature, n_neuron,bias=True)
#        self.layer_out = torch.nn.Linear(n_neuron, out_feature,bias=True)
#        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_neuron, n_neuron, bias=True) for i in range(n_hidden)])
#        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU, 'tanshrink':torch.nn.Tanhshrink}
#        self.activation = self.activations[activation_tag]()
#        self.loss_fn = torch.nn.MSELoss()
#
#    def forward(self,x):
#        x = self.activation(self.layer_in(x))
#        for layer in self.layers:
#            x = self.activation(layer(x))
#        x = self.layer_out(x)
#        return x
#
#    def fit(self, D, lr=0.001, n_iter=1, load=False):
#        x, y = D
#        if load:
#            self.load()
#        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#        for epoch in range(n_iter):
#            optimizer.zero_grad()
#            loss = self.loss_fn(self.forward(x), y)
#            loss.backward()
#            optimizer.step()
#
#    def predict(self, x):
#        return self.forward(x)
#
#    @torch.no_grad()
#    def test(self, D):
#        x, y = D
#        return self.loss_fn(self.forward(x), y)
#
#    def load(self, path='model.pt'):
#        self.load_state_dict(torch.load(path))
