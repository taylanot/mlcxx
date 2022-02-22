import torch 
import torch.nn as nn
from torch.autograd import Variable
from argparse import Namespace
import matplotlib.pyplot as plt

torch.manual_seed(24)
class Simpleton(nn.Module):
    def __init__(self, feature_size, output_size):
        super(Simpleton,self).__init__()
        self.layer = nn.Linear (feature_size, output_size)
    def forward(self, x):
        return self.layer(x)

class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.lmbda= nn.Parameter(torch.ones(1)*0.5)
    def forward(self):
        return self.lmbda


class sample_a():
    def __init__(self, dim, mean, cov):
        self.dim = dim
        self.dist = torch.distributions.MultivariateNormal(torch.ones(dim)*mean, torch.eye(dim)*cov)
    def sample(self, N):
        return self.dist.sample(torch.Size((N,1))).reshape(-1,self.dim)

class sample_data():
    def __init__(self, dim, cov):
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim)*cov)
        self.dim = dim
    def sample(self,a, N):
        x = self.dist.sample(torch.Size((N,1))).reshape(-1, self.dim)
        y = x @ a.T + torch.FloatTensor(N,1).normal_(0,1)
        return x, y
    
def my_config():
    conf = dict()
    conf['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf['dim'] = 1
    conf['mean'], conf['cov'] = 0, 1 
    conf['Ntrn'] = 5
    conf['Ntst'] = 1000
    conf['Nval'] = 5
    conf['network'] = Simpleton(conf['dim'],1)
    conf['network'].to(conf['device'])
    conf['lr'] = 0.001
    conf['lmbda'] = 0.001
    conf['reg'] = Loss()
    conf['reg'].to(conf['device'])
    conf['epoch'] = 5000
    conf['tol'] = 1.e-6
    conf['optimizer'] = torch.optim.SGD(list(conf['network'].parameters())+list(conf['reg'].parameters()),\
                                        conf['lr'], weight_decay=0.)#conf['lmbda'])
    conf['optimizer_wo'] = torch.optim.SGD(list(conf['network'].parameters()),\
                                        conf['lr'], weight_decay=0.)#conf['lmbda'])
    conf['optimizer_reg'] = torch.optim.SGD(list(conf['reg'].parameters()),\
                                        conf['lr'], weight_decay=0.)#conf['lmbda'])
    conf['loss'] = torch.nn.MSELoss()
    return conf

def main(conf):
    conf = Namespace(**conf)
    dist_a = sample_a(conf.dim, conf.mean, conf.cov)
    dist_data = sample_data(conf.dim, conf.cov)
    a = dist_a.sample(1)
    a.to(conf.device)
    xtrn, ytrn = dist_data.sample(a, conf.Ntrn)
    xtst, ytst = dist_data.sample(a,conf.Ntst)
    xval, yval = dist_data.sample(a,conf.Nval)
    xtrn.to(conf.device);ytrn.to(conf.device);
    xtst.to(conf.device);ytst.to(conf.device);
    #print(list(conf.network.parameters()))   
    def train_reg():
        torch.manual_seed(24)
        for layer in conf.network.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        #for i in range(conf.epoch):
        conv = False 
        loss = 0.
        prev = 1000.
        while not conv:
            conf.optimizer.zero_grad()
            l2_reg = sum(torch.linalg.norm(p)**2 for p in conf.network.parameters())
            #params = conf.network.parameters()
            loss = conf.loss(conf.network(xtrn),ytrn)
            total_loss = loss + conf.reg()[0]*l2_reg/2.
            if abs(prev - loss) <= conf.tol:
                break
            #for i, param in enumerate(params):
            #    if i % 2 == 0:
            #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
            #    else:
            #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
            total_loss.backward()
            conf.optimizer.step()
            
            prev = loss
            #print(list(conf.reg.parameters()))
        return test()
    
    def train_reg_correct():
        torch.manual_seed(24)
        for layer in conf.network.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        #for i in range(conf.epoch):
        conv = False 
        loss = 0.
        prev = 1000.
        while not conv:
        #for i in range(1):
            conf.optimizer_wo.zero_grad()
            conf.optimizer_reg.zero_grad()
            l2_reg = sum(torch.linalg.norm(p)**2 for p in conf.network.parameters())
            #params = conf.network.parameters()
            loss = conf.loss(conf.network(xtrn),ytrn)
            total_loss = loss + conf.reg()[0]*l2_reg/2.

            val_loss = conf.loss(conf.network(xval),yval) + conf.reg()[0]*l2_reg/2.
            val_loss.backward()

            conf.optimizer_wo.step()
            conf.optimizer_reg.step()
            if abs(prev - total_loss) <= conf.tol:
                break
            #for i, param in enumerate(params):
            #    if i % 2 == 0:
            #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
            #    else:
            #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
                        
            prev = loss
            #print(list(conf.reg.parameters()))
            print(loss)
        return test()

    def train_reg2():
        torch.manual_seed(24)
        err = []
        for lmbda in torch.linspace(0,2,10):
            for layer in conf.network.children():
               if hasattr(layer, 'reset_parameters'):
                   layer.reset_parameters()
            #for i in range(conf.epoch):
            conv = False 
            loss = 0.
            prev = 1000.
            while not conv:
                conf.optimizer_wo.zero_grad()
                l2_reg = sum(torch.linalg.norm(p)**2 for p in conf.network.parameters())
                #params = conf.network.parameters()
                loss = conf.loss(conf.network(xtrn),ytrn)
                total_loss = loss + lmbda*l2_reg/2.
                if abs(prev - loss) <= conf.tol:
                    break
                #for i, param in enumerate(params):
                #    if i % 2 == 0:
                #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
                #    else:
                #        loss += conf.reg()[0]*torch.linalg.norm(param)**2/2
                total_loss.backward()
                conf.optimizer_wo.step()
                prev = loss
            err.append((test(), lmbda))
        return min(err)


    def train_noreg():
        torch.manual_seed(24)
        for layer in conf.network.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        #for i in range(conf.epoch):
        conv = False 
        loss = 0.
        prev = 1000.
        while not conv:
            conf.optimizer_wo.zero_grad()
            loss = conf.loss(conf.network(xtrn),ytrn) 
            if abs(prev - loss) <= conf.tol:
                break
            loss.backward()
            conf.optimizer_wo.step()
            prev = loss
        return test()

    def get_params():
        print(list(conf.network.parameters()))

    def test():
        with torch.no_grad():
            return conf.loss(conf.network(xtst),ytst)
    
    print('reg_train',train_reg())
    print('reg_fixed',train_reg2())
    print('noreg',train_noreg())
    print('reg_train_correct',train_reg_correct())
    
    #print(list(conf.network.parameters()))
    #print(list(conf.reg.parameters()))
    #x = torch.linspace(-1,1,100).reshape(-1,1) 
    #plt.scatter(xtrn, ytrn)
    #plt.plot(x, conf.network(x).detach().numpy(),color='r')
    #plt.savefig('training.pdf')
        

    
main(my_config())

