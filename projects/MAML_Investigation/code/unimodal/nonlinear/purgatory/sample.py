import torch 
from time import time 
import numpy as np
import matplotlib.pyplot as plt


dim = 50
N = 1000000

#start = time()
#means = torch.ones(N,dim)
#covs = torch.eye(dim)
#covs = covs.repeat(N, 1, 1)
#
#tasks = torch.distributions.MultivariateNormal(means, covs)
#(tasks.sample())
#end = time()
#print(end-start)

#start = time()
#tasks = np.random.multivariate_normal(np.ones(dim), np.eye(dim), N)
#end = time()
#print(end-start)


#start = time()
#tasks = torch.distributions.MultivariateNormal(torch.ones(dim),torch.eye(dim))
#print(torch.mean(tasks.sample(torch.Size([N,1])), axis=0))
#print(torch.mean(tasks.sample(torch.Size([N,1])).squeeze()))
#end = time()
#print(end-start)
#dim = 2
#Ntrn = 2
#Na = 4
#Nz = 3
#a = torch.ones(Na,Nz,Ntrn,dim)
#a[1] *= 2 
#a[2] *= 3 
#a[3] *= 4 
#print(a)
#print(a[1][2])
#b = torch.ones(Na,dim)
#
##for i in range(Na):
##    print( b[i] @ a[i] )
#print(a)
#print(b)
#ys = []
#for i in range(Na):
#    ys.append((a[i]@ b[i]).reshape(Nz, Ntrn,1))
#print(torch.cat(ys).reshape(Na, Nz, Ntrn, 1))

class MAML(torch.nn.Module):
    def __init__(self, in_feature, n_neuron, out_feature, n_hidden=0, activation_tag='softsign'):
        super(MAML, self).__init__()

        self.layer_in = torch.nn.Linear(in_feature, n_neuron,bias=True)
        self.layer_out = torch.nn.Linear(n_neuron, out_feature,bias=True)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_neuron, n_neuron, bias=True) for i in range(n_hidden)])
        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU, 'tanshrink':torch.nn.Tanhshrink}
        self.activation = self.activations[activation_tag]()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self,x):
        x = self.activation(self.layer_in(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.layer_out(x)
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
        self.load_state_dict(torch.load(path))

torch.manual_seed(24) # KOBEEEEE
x = torch.FloatTensor(10,1).uniform_(-5,5)
x = torch.linspace(-5,5,100).reshape(-1,1)
y = torch.sin(x) #+ torch.FloatTensor(x.shape[0],1).normal_(0,0.1)

model = MAML(1,10,1,5)
model.fit((x,y), n_iter=1000)
print(model.test((x,y)))
torch.save(model.state_dict(), 'model.pt')
#print(list(model.parameters()))
model.load()
x_plot = torch.linspace(-5,5, 1000).reshape(-1,1)
plt.plot(x_plot.detach().numpy(), model.predict(x_plot).detach().numpy())
plt.plot(x_plot.detach().numpy(),torch.sin(x_plot).detach().numpy())
plt.savefig('test.pdf')
#print(list(model.parameters()))
print(model.test((x,y)))






