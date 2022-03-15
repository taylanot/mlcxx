import numpy as np
import GPy
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

class sample_D():
    def __init__(self, dim=1, cov=2):
        self.dist = np.random.multivariate_normal
        self.dim = dim
        self.cov = cov
    def sample(self,model, N):
        x = torch.Tensor(self.dist(np.ones(self.dim)*0, np.eye(self.dim)*self.cov, N))
        y = model(x)
        return x, y

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


train_dist = sample_D()
net_train = NonlinearNetwork(1,1)
net_train.load('model.pt')

net = NonlinearNetwork(1,1)
for i in range(10000):
    D = train_dist.sample(net_train, 20)
    net.fit(D)
torch.save(net.state_dict(), 'model2.pt')

xp = np.linspace(-4,4,100).reshape(-1,1)
y_net_train  = net_train.predict(torch.Tensor(xp)).detach().numpy()
y_net = net(torch.Tensor(xp)).detach().numpy()
plt.plot(xp, y_net_train)
plt.plot(xp, y_net)
plt.savefig('check_fit.pdf')
