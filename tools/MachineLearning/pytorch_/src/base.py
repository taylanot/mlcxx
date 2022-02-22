import torch
from statistics import mean,stdev
from torch.utils.data import DataLoader
from .util import ewma
import matplotlib.pyplot as plt 
import time


class network_modules():
    def __init__(self):
        super(network_modules,self).__init__()

    def train_step(self, train_loader):

        """ Training step for one epoch """

        sum_loss = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data               # Decompose train_loader

            self.optimizer.zero_grad()               # Clear gradient history

            outputs = self.forward(inputs)               # Feed-forward
            loss = self.loss_func(outputs,labels)    # Loss
            loss.backward()                     # Backpropagate
            self.optimizer.step()                    # Step towards optimum
            sum_loss += loss                    # Add the losses 

        avg_loss = sum_loss / len(train_loader) # Average the loss
        
        return avg_loss

    def test_step(self, test_loader):
        
        """ Testing step for one epoch """

        sum_loss = 0

        with torch.no_grad():                       # Cancel gradient calculation

            for i, data in enumerate(test_loader):  
                inputs, labels = data               # Decompose data 

                outputs = self.forward(inputs)               # Feed-forward
                loss = self.loss_func(outputs,labels)    # Loss
                sum_loss += loss                    # Add the losses

            avg_loss = sum_loss / len(test_loader)  # Average the loss

            return avg_loss

    def train(self, dataset, epochs=1000, ewma=False):

        """ Tranining Loop """

        if ewma:
            run_avg_train = ewma()
            run_avg_test = ewma()
            for epoch in range(epochs):
                run_avg_train(self.train_step(dataset.train_loader))
                run_avg_test(self.test_step(dataset.test_loader))
            return run_avg_train.get_avg(), run_avg_test.get_avg()
        else:
            train_loss = []
            test_loss = []
            epoch_counter = []

            for epoch in range(epochs):
                train_loss.append(self.train_step(dataset.train_loader).item())
                test_loss.append(self.test_step(dataset.test_loader).item())
                epoch_counter.append(epoch)
            return epoch_counter, train_loss, test_loss


    def xval(self, dataset, epochs=1,return_dictionary=False, reset='init'):
        
        """ Cross-validation loop """

        train_stats , test_stats = self.xval_loader(dataset.k_loaders,epochs=epochs, reset=reset)

                
        if return_dictionary:
            return { 'mean_train':train_stats[0],'std_train':train_stats[1],'mean_test':test_stats[0],'std_test':test_stats[1]}

        else:
            return train_stats, test_stats


    def xval_loader(self, loaders, epochs=1, reset='init'):
        
        """ Cross-validation loaders """

        test_score = []
        train_score = []

        for i,fold in enumerate(loaders):

            print('xval-fold->'+str(i+1),)
            start = time.perf_counter()

            train_loader = fold[0]
            test_loader = fold[1]

            if reset == 'init':
                self._reset_linear_layers()

            elif reset == 'load':
                self._reset_load_parameters()

            #self.apply(self._weight_init_linear)
            #self.apply(self._bias_init_linear)

            run_avg_train = ewma()
            run_avg_test = ewma()

            for epoch in range(epochs):
                run_avg_train(self.train_step(train_loader).item())
                run_avg_test(self.test_step(test_loader).item())

            test_score.append(mean(run_avg_train.get_avg()))
            train_score.append(mean(run_avg_test.get_avg()))
            end = time.perf_counter()
            print('Elapsed-time:',end-start)

        

        train_stats = [mean(train_score), stdev(train_score)]
        test_stats = [mean(test_score), stdev(test_score)]
        
        return train_stats, test_stats



    def lcurve(self, dataset, epochs=1000, return_dictionary=False):
        
        """ Learning Curve loop """

        test_stats = []
        train_stats = []
        train_sizes = dataset.train_sizes

        for sets in dataset.learning_loaders:
            train, test = self.xval_loader(sets, epochs)
            train_stats.append(train), test_stats.append(test)

        if return_dictionary:
            mean_train = [train_stats[i][0] for i in range(len(train_stats))]
            mean_test = [test_stats[i][0] for i in range(len(test_stats))]

            std_train = [train_stats[i][1] for i in range(len(train_stats))]
            std_test = [test_stats[i][1] for i in range(len(test_stats))]

            return { 'sizes': train_sizes, 'mean_train':mean_train,'std_train':std_train,'mean_test':mean_test,'std_test':std_test}

        else:
            return train_sizes, train_stats, test_stats


    



##########################################################################################
# PURGATORY
##########################################################################################

def train(train_loader, net, optimizer, loss_func):

    """ Training loop """

    sum_loss = 0

    for i, data in enumerate(train_loader):
        #print(i)
        inputs, labels = data               # Decompose train_loader

        optimizer.zero_grad()               # Clear gradient history

        outputs = net(inputs)               # Feed-forward
        loss = loss_func(outputs,labels)    # Loss
        loss.backward()                     # Backpropagate
        optimizer.step()                    # Step towards optimum
        sum_loss += loss                    # Add the losses 

    avg_loss = sum_loss / len(train_loader) # Average the loss
    
    return avg_loss

def test(test_loader, net, loss_func):
    
    """ Testing loop """

    sum_loss = 0

    with torch.no_grad():                       # Cancel gradient calculation

        for i, data in enumerate(test_loader):  
            inputs, labels = data               # Decompose data 

            outputs = net(inputs)               # Feed-forward
            loss = loss_func(outputs,labels)    # Loss
            sum_loss += loss                    # Add the losses
            #print(sum_loss)

        avg_loss = sum_loss / len(test_loader)  # Average the loss

        return avg_loss

def xval(dataset, net, optimizer, loss_func, epochs=1000):
    
    """ Cross-validation loop """

    test_score = []
    train_score = []

    for i,fold in enumerate(dataset.k_loaders):

        print('xval-fold->'+str(i+1),)
        start = time.perf_counter()

        train_loader = fold[0]
        test_loader = fold[1]

        net.apply(net._weight_init_linear)
        net.apply(net._bias_init_linear)

        test_loss = 0. 
        train_loss = 0. 
        run_avg_train = ewma()
        run_avg_test = ewma()
        check_1 = []
        check_2 = []

        epoch = 1 
        for _ in range(epochs):
            run_avg_train(train(train_loader, net, optimizer, loss_func).item())
            run_avg_test(test(test_loader, net, loss_func).item())
            #check_1.append(train(train_loader, net, optimizer, loss_func).item())
            #check_2.append(test(test_loader, net, loss_func).item())
            epoch += 1

        test_score.append(mean(run_avg_train.get_avg()))
        train_score.append(mean(run_avg_test.get_avg()))
        end = time.perf_counter()
        print('Elapsed-time:',end-start)

    
        #plt.plot(range(epochs),run_avg_train.get_avg(True),label='train'+str(i))
        #plt.plot(range(epochs),run_avg_test.get_avg(True),label='test'+str(i))
        #plt.plot(range(epochs)[100:1000],check_1[100:1000],label='train'+str(i))
        #plt.plot(range(epochs)[100:1000],check_2[100:1000],label='test'+str(i))

    train_stats = [mean(train_score), stdev(train_score)]
    test_stats = [mean(test_score), stdev(test_score)]
    
    return train_stats, test_stats

        



        

        








