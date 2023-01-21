import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import matplotlib.pyplot as plt
np.random.seed(24)
torch.manual_seed(100)
################################################################################
def funcy(x, noise = True): # Create Data
    val = x**2 
    dist = torch.distributions.beta.Beta(torch.tensor([2.]),
                                         torch.tensor([5.]))
    eps = dist.sample(x.size()).reshape(-1,1)
    if noise:
        return  val + eps
    else:
        return val
def funcy2(x, noise = True): # Create Data
    val = x**2 
    if noise:
        return  val + torch.randn_like(val)
    else:
        return val

################################################################################

Ntrn = 100
Ntst = 1000
nposteriorsamples = 100

Xtrn = torch.normal(5,5,(Ntrn,1))                      # Training Data Feature
Xtst = torch.linspace(0,2,Ntst).reshape(-1,1)
Ytrn = funcy2(Xtrn)                                          # Training Data Labels

Xtst = torch.normal(5,5,(Ntst,1))
Xtst,_ = torch.sort(Xtst)
Xtst = torch.linspace(0,2,Ntst).reshape(-1,1)

Ytst = funcy2(Xtst)


# Initialize model
model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                      in_features=1, out_features=100),
                      nn.ReLU(),
                      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                      in_features=100, out_features=1))

def train(model,x, y, epochs=100):
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        pre = model(x)
        mse = mse_loss(pre, y)
        kl = kl_loss(model)
        cost = mse + kl_weight*kl
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch%100 == 0:
            print("MSE:{}-KL:{}".format(mse, kl))
        #if epoch%100 == 0:
        #    print("MSE:{}".format(mse))

train(model, Xtrn, Ytrn, 10000)

plt.scatter(Xtst, Ytst)
for i in range(nposteriorsamples):
    plt.plot(Xtst, model(Xtst).detach().numpy(),alpha=0.3)
plt.savefig("pred.pdf")

plt.figure()
ys = []
for i in range(100000):
    ys.append( model(torch.ones(1,1)*100).detach().numpy())
ys = np.array(ys).flatten()
plt.hist(ys)
print(np.std(ys))
plt.savefig("hist.pdf")
