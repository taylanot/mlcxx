import torch
import torch.nn.functional as F
from MLtorch.src.dataset import *
from MLtorch.src.generate_data import *
from MLtorch.src.base import * 
from torchsummary import summary
import copy 
import os 
import matplotlib.pyplot as plt 

class MAML():
    def __init__(self, network, alpha, beta, task_dist, num_points_task, num_task_sample, first_order = False, show_weights=False, plot=False):
        self.network = network
        self.weights = self.network.list_parameters()
        self.alpha = alpha
        self.beta = beta
        self.pT = task_dist
        self.k = num_points_task
        self.num_task = num_task_sample
        self.loss = [ ]
        self.optimizer = torch.optim.Adam(self.weights, self.beta)
        self.loss_func = torch.nn.MSELoss()
        self.fo = first_order 
        self.print_step = 10
        self.plot_step = 100
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
            gradients = torch.autograd.grad(loss,self.weights, create_graph=False)
        else:
            gradients = torch.autograd.grad(loss,self.weights, create_graph=True)

        temp_weights = [w - self.alpha*grad for w, grad in zip(temp_weights, gradients)]

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

            if self.plot:
                if epoch % self.plot_step == 0:
                    self.plot(epoch,path=path)
            
    def train(self, epochs, path='results'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs,path)
        return self.loss

    def plot(self, epoch, res=100, path='results'):
        xp = torch.linspace(self.pT.domain_bounds[0],self.pT.domain_bounds[1],res).reshape(-1,1)
        boundary_tasks = self.pT.task_def.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='blue',label='prediction')
        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(epoch)+'.pdf')
        plt.close()


class ConvexMAML():
    def __init__(self, network, alpha, beta, task_dist, num_points_task, num_task_sample, first_order = False, show_weights=True):
        self.network = network
        self.weights = self.network.list_parameters()
        self.alpha = alpha
        self.beta = beta
        self.pT = task_dist
        self.k = num_points_task
        self.num_task = num_task_sample
        self.loss = [ ]
        self.optimizer = torch.optim.Adam(self.weights, self.beta)
        self.loss_func = torch.nn.MSELoss()
        self.fo = first_order 
        self.print_step = 10
        self.plot_step = 50 
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
            gradients = torch.autograd.grad(loss,self.weights, create_graph=False)
        else:
            gradients = torch.autograd.grad(loss,self.weights, create_graph=True)

        temp_weights = [w - self.alpha*grad for w, grad in zip(temp_weights, gradients)]
        #temp_weights = torch.nn.Parameter(self.network.analytic(D_train))

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

            if epoch % self.plot_step == 0:
                self.plot(epoch,path=path)
            
    def train(self, epochs, path='results'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs,path)
        return self.loss

    def plot(self, epoch, res=100, path='results'):
        xp = torch.linspace(self.pT.domain_bounds[0],self.pT.domain_bounds[1],res).reshape(-1,1)
        boundary_tasks = self.pT.task_def.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='blue',label='prediction')
        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(epoch)+'.pdf')
        plt.close()

class multitask():
    def __init__(self, network, beta, task_dist, num_points_task, num_task_sample, show_weights=False):
        self.network = network
        self.pT = task_dist
        self.k = num_points_task
        self.num_task = num_task_sample
        self.loss = [ ]
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.beta)
        self.loss_func = torch.nn.MSELoss()
        self.print_step = 10
        self.plot_step = 50
        self.show_weights = show_weights

    def inner(self, task):
        D_train = self.pT.sample_data(task, size=self.k)
        D_test = self.pT.sample_data(task, size=self.k)

        x_train, y_train = D_train;
        x_test, y_test = D_test;  

        pred_train = self.network(x_train)
        train_loss = self.loss_func(pred_train,y_train) 

        pred_test = self.network(x_test)
        
        test_loss = self.loss_func(pred_test, y_test)

        return train_loss, test_loss

    def outer(self, epochs, path):

        for epoch in range(epochs+1):
            train_loss = 0
            test_loss = 0
            self.optimizer.zero_grad()
            for task in range(self.num_task):
                task_i = self.pT.sample_task()
                losses = self.inner(task_i) 
                train_loss += losses[0] / self.num_task
                test_loss += losses[1] / self.num_task

            train_loss.backward()
            self.optimizer.step()
            self.loss.append(train_loss.item())
            if epoch % self.print_step == 0:
                print("epoch:{}->loss:{}".format(epoch, test_loss.item()))
                if self.show_weights:
                    print("weights:{}".format(list(self.network.parameters())))

            if epoch % self.plot_step == 0:
                self.plot(epoch,path=path)

            

    def train(self, epochs, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs, path)
        return self.loss

    def plot(self, epoch, res=100, path='results'):
        xp = torch.linspace(self.pT.domain_bounds[0],self.pT.domain_bounds[1],res).reshape(-1,1)
        boundary_tasks = self.pT.task_def.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='blue',label='prediction')
        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(epoch)+'.pdf')
        plt.close()

class multitaskmax():
    def __init__(self, network, beta, task_dist, num_points_task, num_task_sample, show_weights=True):
        self.network = network
        self.pT = task_dist
        self.k = num_points_task
        self.num_task = num_task_sample
        self.loss = [ ]
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.beta)
        self.loss_func = torch.nn.MSELoss()
        self.print_step = 10
        self.plot_step = 20
        self.show_weights = show_weights

    def inner(self, task):
        D_train = self.pT.sample_data(task, size=self.k)
        D_test = self.pT.sample_data(task, size=self.k)

        x_train, y_train = D_train;
        x_test, y_test = D_test;  

        pred_train = self.network(x_train)
        train_loss = self.loss_func(pred_train,y_train) 

        pred_test = self.network(x_test)
        
        test_loss = self.loss_func(pred_test, y_test)

        return train_loss, test_loss

    def outer(self, epochs, path):

        for epoch in range(epochs):
            train_loss = []
            test_loss = []
            self.optimizer.zero_grad()
            for task in range(self.num_task):
                task_i = self.pT.sample_task()
                losses = self.inner(task_i) 
                train_loss.append(losses[0])
                test_loss.append(losses[1])


            max_loss = max(test_loss)

            max_index = test_loss.index(max_loss)

            loss = test_loss[max_index]

            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.item())
            if epoch % self.print_step == 0:
                print("epoch:{}->loss:{}".format(epoch, loss.item()))
                if self.show_weights:
                    print("weights:{}".format(list(self.network.parameters())))

            if epoch % self.plot_step == 0:
                self.plot(epoch,path=path)

            

    def train(self, epochs, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs, path)
        return self.loss

    def plot(self, epoch, res=100, path='results'):
        xp = torch.linspace(self.pT.domain_bounds[0],self.pT.domain_bounds[1],res).reshape(-1,1)
        boundary_tasks = self.pT.task_def.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='blue',label='prediction')
        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(epoch)+'.pdf')
        plt.close()

class model_average():
    def __init__(self, network, beta, task_dist, num_points_task, show_weights=True):
        self.network = network
        self.networks = [copy.deepcopy(network), copy.deepcopy(network)]

        self.pT = task_dist
        self.k = num_points_task
        self.loss = [ ]
        self.beta = beta
        self.optimizers = [torch.optim.Adam(self.networks[0].parameters(), self.beta), torch.optim.Adam(self.networks[1].parameters(), self.beta)]
        self.loss_func = torch.nn.MSELoss()
        self.print_step = 10
        self.plot_step = 20
        self.show_weights = show_weights

    def inner(self, task, model_id):
        D_train = self.pT.sample_data(task, size=self.k)
        D_test = self.pT.sample_data(task, size=self.k)

        x_train, y_train = D_train;
        x_test, y_test = D_test;  

        pred_train = self.networks[model_id](x_train)
        train_loss = self.loss_func(pred_train,y_train) 

        pred_test = self.networks[model_id](x_test)
        
        test_loss = self.loss_func(pred_test, y_test)

        return train_loss, test_loss

    def outer(self, epochs, path):

        for model_id in range(2):
            train_loss = []
            test_loss = []
            for epoch in range(epochs):
                self.optimizers[model_id].zero_grad()
                task_i = self.pT.task_def.boundary_tasks[model_id]
                losses = self.inner(task_i, model_id) 
                train_loss.append(losses[0])
                test_loss.append(losses[1])

                loss = losses[0]

                loss.backward()
                self.optimizers[model_id].step()
                self.loss.append(loss.item())
                if epoch % self.print_step == 0:
                    self.average_models()
                    print("model:{}->epoch:{}->loss:{}".format(model_id, epoch, loss.item()))
                    if self.show_weights:
                        print("weights:{}".format(list(self.network.parameters())))

                if epoch % self.plot_step == 0:
                    self.plot(epoch,path=path,model=model_id)

    
    def average_models(self):
        state_dicts = [self.networks[i].state_dict() for i in range(2)]
        state_dict_base = self.network.state_dict()

        # Average all parameters
        for key in state_dict_base:
            state_dict_base[key] = (state_dicts[0][key] + state_dicts[1][key]) / 2.

        self.network.load_state_dict(state_dict_base)



    def train(self, epochs, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.outer(epochs, path)
        return self.loss

    def plot(self, epoch, res=100, path='results', model=0):
        xp = torch.linspace(self.pT.domain_bounds[0],self.pT.domain_bounds[1],res).reshape(-1,1)
        boundary_tasks = self.pT.task_def.boundaries(xp)

        plt.figure()
        plt.plot(xp, self.network(xp).detach().numpy(),color='blue',label='prediction')
        plt.plot(xp, boundary_tasks[0],'--',color='red',label='bounds')
        plt.plot(xp, boundary_tasks[1], '--',color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(str(path)+'/'+str(model)+'_'+str(epoch)+'.pdf')
        plt.close()

        




            


        

        
        


        
        
