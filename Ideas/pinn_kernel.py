import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(24)

def pairwise_l2_distance(x,y):
    D = -2 * x @  y.T + torch.sum(y**2, axis=1) + torch.sum(x**2, axis=1)[:,None]
    D[D<0] = 0.
    return D

class RBF():
    def __init__(self,l=1):
        self.l = l
    def __call__(self, x, xp=None):
        if np.any(xp == None):
            xp = x
        return torch.exp(-0.5 * pairwise_l2_distance(x,xp)/self.l**2)

def grad(f,x,t):
  gradients = []
  for order in range(1,t+1):
    if order == 1:
      f = f.unsqueeze(0)
    else:
      f = gradients[-1]
    dx = torch.zeros_like(f)
    gradient = []
    for i in range(dx.shape[0]):
      for j in range(dx.shape[-1]):
        dx[i,:,j]=1.
        df,= torch.autograd.grad(f[i],x,dx[i],retain_graph=True,create_graph=True, allow_unused=True)
        dx[i,:,j]=0.
        gradient.append(df)
    if all(df is None for df in gradient):
      gradients.append(None)
    else:
      gradients.append(torch.stack(gradient))
  return gradients

class PINN(torch.nn.Module):
    def __init__(self, n_hidden ):
        super(PINN, self).__init__()

        self.input = torch.nn.Linear(1,10)
        self.output = torch.nn.Linear(10,1)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(10,10) for i in range(n_hidden)])

        self.activation = torch.nn.Softsign()

    def forward(self,x):
        x = self.activation(self.input(x))
        for layer in self.hidden:
          x = layer(x)
        return self.output(x)

class PIKR(torch.nn.Module):
    def __init__(self, X, kernel, lmbda=0, lr=0.1):
        super(PIKR, self).__init__()
        self.X = X
        N, d = X.shape[0], X.shape[1]
        self.lmbda= lmbda
        self.kernel = kernel
        self.lr = lr
        self.alpha = torch.nn.Parameter(torch.randn(N, 1))
        self.alpha.requires_grad = True

    def forward(self, X):
        return (self.kernel(X, self.X)) @ self.alpha

def loss_PDE(model, x):
    g = x.clone()
    g.requires_grad = True
    u = model(g)
    grads = grad(u,g,2)
    u_x, u_xx = grads[0][0],grads[1][0]
    f = u_xx + torch.ones_like(u_xx) 
    return loss_func(f,torch.zeros_like(f))
  
def loss_BC(model, x):
    return loss_func(model(x),torch.zeros_like(x))

def train(model, xb, x, epochs):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-1)
    for epoch in range(epochs):
      optimizer.zero_grad()

      loss = loss_BC(model, xb) + loss_PDE(model, x)

      loss.backward()

      optimizer.step()

      print(loss)


x_domain = torch.linspace(0.2,0.8,10).reshape(-1,1)
x_dbc = torch.tensor([0.,1.]).reshape(-1,1)

#model = PINN(2)
model = PIKR(x_domain, RBF(1))
loss_func = torch.nn.MSELoss()
train(model, x_dbc, x_domain,3000)
#train_kernel(model1, model2, x_dbc, x_domain,1000)



#poisson.train(x_dbc, x_domain, 1000)

import matplotlib.pyplot as plt
x_domain = torch.linspace(0.,1.,100).reshape(-1,1)
plt.plot(x_domain.detach().numpy(), (model(x_domain)).detach().numpy())
plt.plot(x_domain.detach().numpy(), (-0.5*x_domain**2+0.5*x_domain).detach().numpy())
plt.savefig('pinn.pdf')
#N = 10
#f = lambda x: torch.sin(x) #+ torch.FloatTensor(x.shape[0],1).normal_(0,0.1)
#x = torch.FloatTensor(N,1).normal_(0,5)
#x.requires_grad=True
#D = (x,f(x))
#kernel = RBF()
#network = KernelRidgeGD(D, 0., kernel)
#optimizer = torch.optim.SGD(network.parameters(), 0.1)
#loss_fn = torch.nn.MSELoss()
#for i in range(100):
#    optimizer.zero_grad()
#    loss = loss_fn(network(x), f(x))
#    loss.backward()
#    optimizer.step()
#xp = torch.linspace(-10,10,100).reshape(-1,1)
#plt.plot(xp, network(xp).detach().numpy(),'-.',color='teal',label='pred-analy')
#plt.scatter(x.detach().numpy(), f(x).detach().numpy(), color='r')
#plt.plot(xp.detach().numpy(), torch.sin(xp).detach().numpy(), color='k')
###plt.legend()
#plt.savefig('taytay.pdf')
##
#
#
#
#
##    def fit(self, D, n_iter):
##        X, Y = D
##        N, d = X.shape[0], X.shape[1]
##        self.alpha = torch.FloatTensor(N,d).normal_(0,1)
##        self.X = X
##        kern = self.kernel(X,X)
##        for i in range(n_iter):
##            grad = 2 * kern @ ( -Y + kern @ (self.alpha))
##            self.alpha = self.alpha - self.lr * grad
##
##    def __call__(self, X):
##        return (self.kernel(X, self.X)) @ self.alpha
#
#
