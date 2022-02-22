import torch
from collections import OrderedDict

class NonlinearNetwork(torch.nn.Module):
    def __init__(self, in_feature, out_feature, n_hidden=1,  n_neuron=40,\
            activation_tag='tanh'):
        super(NonlinearNetwork, self).__init__()
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh,\
                'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU,\
                'tanshrink':torch.nn.Tanhshrink}
        self.activation = self.activations[activation_tag]
        keys = ['hidden-'+str(i+1) for i in range(n_hidden)]
        vals = [torch.nn.Linear(n_neuron, n_neuron) for i in range(n_hidden)]
        fixed_layers = {'input':torch.nn.Linear(in_feature, n_neuron),
                        'output':torch.nn.Linear(n_neuron, out_feature)}
        layer_info = OrderedDict(zip(keys,vals))
        layer_info.update(fixed_layers)
        layer_info.move_to_end('input', last=False)

        self.layers = torch.nn.Sequential(layer_info)

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

#model = NonlinearNetwork(1,1)
#print(model)
