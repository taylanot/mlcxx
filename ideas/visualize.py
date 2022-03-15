import torch 
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

class NonlinearNetwork(torch.nn.Module):
    def __init__(self, in_feature, out_feature, n_hidden=2,  n_neuron=40,\
            activation_tag='relu'):
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


network = NonlinearNetwork(1,1)
network.load('model.pt')
params = list(network.parameters())
#xp = torch.linspace(-5, 5,100).reshape(-1,1)
#plt.plot(xp, network(xp).detach().numpy())
#print(param)
#a = torch.Tensor(10, 40).uniform_(0, 1) 
size = 2 
fig, axs = plt.subplots(size)
hids = []
for i, param in enumerate(params):
    print(param.detach().numpy().shape == (40,40))
    if param.detach().numpy().shape == (40,40):
        hids.append(param.detach().numpy())
        
data1 = hids[0]
data2 = hids[1]
# find minimum of minima & maximum of maxima
minmin = np.min([np.min(data1), np.min(data2)])
maxmax = np.max([np.max(data1), np.max(data2)])

im1 = axs[0].imshow(data1, vmin=minmin, vmax=maxmax)
im2 = axs[1].imshow(data2, vmin=minmin, vmax=maxmax)

# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)
plt.savefig('visualize.pdf')

network = NonlinearNetwork(1,1)
network.load('multitask.pt')
print(network(torch.Tensor([1])))
params = list(network.parameters())
#xp = torch.linspace(-5, 5,100).reshape(-1,1)
#plt.plot(xp, network(xp).detach().numpy())
#print(param)
#a = torch.Tensor(10, 40).uniform_(0, 1) 
size = 2 
fig, axs = plt.subplots(size)
hids = []
for i, param in enumerate(params):
    print(param.detach().numpy().shape == (40,40))
    if param.detach().numpy().shape == (40,40):
        hids.append(param.detach().numpy())
        
data1 = hids[0]
data2 = hids[1]
# find minimum of minima & maximum of maxima
minmin = np.min([np.min(data1), np.min(data2)])
maxmax = np.max([np.max(data1), np.max(data2)])

im1 = axs[0].imshow(data1, vmin=minmin, vmax=maxmax)
im2 = axs[1].imshow(data2, vmin=minmin, vmax=maxmax)

# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)
plt.savefig('visualize2.pdf')

plt.figure()
xp = torch.linspace(-5, 5,100).reshape(-1,1)
network = NonlinearNetwork(1,1)
network.load('model.pt')
plt.plot(xp, network(xp).detach().numpy())
network = NonlinearNetwork(1,1)
network.load('multitask.pt')
plt.plot(xp, network(xp).detach().numpy())
plt.savefig('pred.pdf')
#print(param)
#a = torch.Tensor(10, 40).uniform_(0, 1) 

