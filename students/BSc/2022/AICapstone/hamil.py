import torch
import hamiltorch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

hamiltorch.set_random_seed(24)
#np.random.seed(24)
torch.manual_seed(24)
################################################################################
def funcy(x, noise = True): # Create Data
    val = torch.sin(x)
    dist = torch.distributions.beta.Beta(torch.tensor([2.]),
                                         torch.tensor([5.]))
    eps = dist.sample(x.size()).reshape(-1,1)
    if noise:
        return  val + eps
    else:
        return val
def funcy2(x):
    return torch.sin(x) + 0.1*torch.normal(0,1,(x.shape[0],1))

################################################################################
device = "cpu"
Ntrn = 100
Ntst = 1000
#nposteriorsamples = 100

Xtrn = torch.normal(5,5,(Ntrn,1))                      # Training Data Feature
Xtst = torch.linspace(-5,5,Ntst).reshape(-1,1)
Ytrn = funcy(Xtrn)                                          # Training Data Labels

Xtst = torch.normal(-5,5,(Ntst,1))
Xtst,_ = torch.sort(Xtst)
Xtst = torch.linspace(-5,5,Ntst).reshape(-1,1)

Ytst = funcy(Xtst)

class Net(nn.Module):

    def __init__(self, layer_sizes, loss = 'multi_class', bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
#         for l in range(len(layer_sizes[:-1])):
#         self.layer_list.append(
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1],bias=True)
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2],bias = self.bias)
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3],bias = self.bias)
#         self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4],bias = self.bias)

    def forward(self, x):
#         for layer in self.layer_list[:-1]:
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        
        return x

layer_sizes = [1,10,10,1]
net = Net(layer_sizes, loss='regression').to(device)

params_init = hamiltorch.util.flatten(net).to(device).clone()
print('Parameter size: ',params_init.shape[0])

tau_list = []
tau = 0.1
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

step_size = 0.001
num_samples = 10000
L = 20
tau_out = 0.1 

net = Net(layer_sizes, loss='regression')
params_init = hamiltorch.util.flatten(net).to(device).clone()
print('Parameter size: ',params_init.shape[0])

params_hmc = hamiltorch.sample_model(net, Xtrn, Ytrn, model_loss='regression',params_init=params_init, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L,tau_out=tau_out,normalizing_const=Ntrn, tau_list=tau_list)
x = torch.Tensor([10.])
x = Xtst[0]
y = Ytst[0]
pred_list, log_prob_list = hamiltorch.predict_model(net, x=x, y=y, model_loss='regression', samples=params_hmc[:], tau_out=tau_out, tau_list=tau_list)

#print(tau_list[0])
#print(tau_out)
#print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
#print('\nExpected MSE: {:.2f}'.format(((pred_list.mean(0) - Ytst)**2).mean()))
burn = 10
plt.hist(pred_list.detach().numpy()[burn:])
plt.savefig("hist.pdf")
