# Internal Stuff
import torch
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import numpy as np
import tikzplotlib
# Define your plot cycle color
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

class MAML():
    def __init__(self, network, alpha, beta, task_dist, num_points_task,\
            num_task_sample, first_order = False, show_weights=False,\
            plot=False):

        self.network = network
        self.weights = list(self.network.parameters())
        self.alpha = alpha
        self.beta = beta
        self.pT = task_dist
        self.k = num_points_task
        self.num_task = num_task_sample
        self.loss = [ ]
        self.optimizer = torch.optim.Adam(self.weights, self.beta)
        self.loss_func = torch.nn.MSELoss()
        self.fo = first_order 
        self.print_step = 1000
        self.plot_step = 10000
        self.plot_bool = plot
        self.show_weights = show_weights

    def inner(self, task):
        temp_weights = [w.clone() for w in self.weights]
        D_train = self.pT.sample_data(task, size=self.k)
        D_test = self.pT.sample_data(task, size=self.k)

        x_train, y_train = D_train;
        x_test, y_test = D_test;  

        pred_train = self.network(x_train,temp_weights)
        loss = self.loss_func(pred_train,y_train) 

        if self.fo:
            gradients = torch.autograd.grad(loss,self.weights,\
                    create_graph=False)
        else:
            gradients = torch.autograd.grad(loss,self.weights,\
                    create_graph=True)

        temp_weights = [w - self.alpha*grad for w, grad in\
                zip(temp_weights, gradients)]

        pred_test = self.network(x_test, temp_weights)
        
        test_loss = self.loss_func(pred_test, y_test)

        return test_loss

    def outer(self, epochs, path):
        
        for epoch in range(1,epochs+1):
            total_loss = 0
            for task in range(self.num_task):
                task_i = self.pT.sample_task()
                loss = self.inner(task_i)
            loss /= self.num_task
            gradients_gradients = torch.autograd.grad(loss,self.weights)

            for w, grad in zip(self.weights, gradients_gradients):
                w.grad = grad

            self.optimizer.step()
            self.loss.append(loss.item())
            if epoch % self.print_step == 0:
                print("epoch:{}->loss:{}".format(epoch, loss.item()))
                if self.show_weights:
                    print("weights:{}".format(list(self.network.parameters())))

            if self.plot_bool:
                if epoch % self.plot_step == 0:
                    self.plot(epoch,path=path)
            
    def train(self, epochs, path='results'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs,path)
        return self.loss

    def plot(self, epoch, res=100, path='results'):
        xp = torch.linspace(self.pT.x_bound[0],self.pT.x_bound[1]\
                ,res).reshape(-1,1)
        boundary_tasks = self.pT.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='black',\
                label='prediction')

        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(epoch)+'.pdf')
        tikzplotlib.save(str(path)+'/'+str(epoch)+'.tex')
        plt.close()


