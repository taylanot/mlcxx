import torch
from .util import read_pickle, tensorize_DataFrame
from .scaler import *
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, SubsetRandomSampler, SequentialSampler 
import itertools as it
import random 


class BaseDataset():

    """ Base class for the ML activities with pytorch """
    
    def __init__(self, dataset=None,  filter_nan=True, **kwargs):

        """ Initialize """

        self.kwargs = kwargs

        self.filter_nan = filter_nan
        self.dataset = dataset


    def set_up(self):

        """ Method: Set up the dataset with little manipulation """

        ################################
        # If you want to read data from file 
        ################################
        if 'infile' and 'outfile' in self.kwargs:
            self.input = tensorize_DataFrame(read_pickle(self.kwargs['infile'])).T
            self.output = tensorize_DataFrame(read_pickle(self.kwargs['outfile'])).T

        ################################
        # If you want to read data from from an DATA object and 
        # it has a tensor attribute
        ################################
        elif 'indata' and 'outdata' in self.kwargs:
            self.input = self.kwargs['indata'].tensor
            self.output = self.kwargs['outdata'].tensor

        ################################
        # If you input TensorDataset 
        ################################
        elif self.dataset != None:
            self.input = self.dataset[:][0]
            self.output = self.dataset[:][1]

        ################################
        # Filter NaN Values
        ################################
        if self.filter_nan:
            self.cleanup_nan()

        ################################
        # Crop data
        ################################

        if 'inx' and 'outx' in self.kwargs:
            self.crop_merge(self.kwargs['inx'], self.kwargs['outx'])

        ################################
        # Scale data
        ################################
        if 'scale_input' in self.kwargs:
            self.unscaled_input = self.input
            self.scale_input(method=self.kwargs['scale_input'])
        if 'scale_output' in self.kwargs:
            self.unscaled_output = self.output
            self.scale_output(method=self.kwargs['scale_output'])

        ################################
        # Initialize your sizes 
        ################################
        self.size = len(self.input)
        self.in_size = self.input.shape[1]
        self.out_size = self.output.shape[1]
        
        self.data = TensorDataset(self.input, self.output)

    def crop_merge(self, inx=None, outx=None):

        """ Method: Merge your data with labels """  
        
        ################################
        # Create Tensor Dataset with desired features and labels
        ################################
        if inx != None and outx != None:
            self.in_size, self.out_size = len(inx), len(outx)
            self.input = self.input[:,inx]
            self.output = self.output[:,outx]


    def cleanup_nan(self):
        
        """ Method: Get rid of NaN values that you might have in your data """

        self.input = self.input[~torch.any(self.output.isnan(),dim=1)]
        self.output = self.output[~torch.any(self.output.isnan(),dim=1)]

    def scale_input(self,method, mode='transform'):
        
        """ Method: Scale inputs """

        if not hasattr(self,'stdzr_inp'):
            self.stdzr_inp = Scaler(self.input,method=method)
        self.input = self.stdzr_inp(self.input, mode)
        self.data = torch.utils.data.TensorDataset(self.input,self.output)

    def scale_output(self,method, mode='transform'):
        
        """ Method: Scale inputs """

        if not hasattr(self,'stdzr_out'):
            self.stdzr_out = Scaler(self.output,method=method)
        self.output = self.stdzr_out(self.output, mode)
        self.data = torch.utils.data.TensorDataset(self.input,self.output)


class SingleTaskDataset(BaseDataset):
    def __init__(self, dataset=None, filter_nan=True, **kwargs):
        super(SingleTaskDataset, self).__init__(dataset,filter_nan, **kwargs)

        self.set_up()

    def split_shuffle(self, test_ratio=0.2, train_size=None, batch_size=128, shuffle=True, spit=False):

        """ Method: Split your data to test and train sets """  

        assert test_ratio == None or train_size == None, "either test_ratio or train_size should be None!"

        if train_size == None:

            test_size = int(test_ratio * self.size)
            train_size = self.size - test_size

        else:

            test_size = self.size - train_size


        self.train_indices, self.test_indices = random_split(self.data,[train_size, test_size])

        self.train_sampler = SubsetRandomSampler(self.train_indices.indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices.indices)

        if spit:
            return train_size, DataLoader(self.data, batch_size=batch_size, sampler=self.train_sampler), DataLoader(self.data, batch_size=batch_size, sampler=self.test_sampler)
        else:
            self.train_loader = DataLoader(self.data, batch_size=batch_size, sampler=self.train_sampler)
            self.test_loader = DataLoader(self.data, batch_size=batch_size, sampler=self.test_sampler)

    def k_fold_split(self, k_fold=5, batch_size=128, shuffle=True, train_size=None, spit=False):

        """ Method: Split your data to for k-fold x-validation """

        fraction = 1./k_fold
        seg = int(self.size * fraction)
        split_sizes = seg * torch.ones(k_fold,dtype=int)
        left_over = self.size - (seg * k_fold)
        count = it.count()
        while left_over != 0:
            split_sizes[count] += 1; left_ver -= 1

        kset = random_split(self.data,split_sizes)
        k_loaders = []
        for i in range(len(kset)):
            remove = kset.pop(i)
            train_indices = []
            [train_indices.append(kset[j].indices) for j in range(len(kset))]
            train_indices = list(it.chain.from_iterable(train_indices))
            test_indices = remove.indices
            if train_size == None:
                train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, sampler=train_sampler)
            else:
                train_subset = torch.utils.data.Subset(self.data, random.sample(train_indices,train_size))
                train_loader = torch.utils.data.DataLoader(train_subset, batch_size=10000)

            test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
            test_loader = torch.utils.data.DataLoader(self.data, batch_size=10000, sampler=test_sampler)
            k_loaders.append((train_loader, test_loader))
            kset.insert(i,remove)
        if spit:
            return k_loaders
        else:
            self.k_loaders = k_loaders

    def learning_curve_split(self, train_sizes, batch_size=128, shuffle=True, cv=5):

        self.train_sizes = train_sizes
        
        self.learning_loaders = [self.k_fold_split(k_fold=cv, batch_size=batch_size, shuffle=shuffle, train_size=size, spit=True) for size in self.train_sizes]

    def select_subset(self, n_samples, batch_size=128, shuffle=True):

        """ Method: Selecting a subset of data for training """

        assert isinstance(n_samples,int), 'Number of samples should be integer mate!'
        self.train_subset = Subset(self.train_dataset, range(n_samples))
        self.train_loader = DataLoader(self.train_subset.dataset, batch_size=batch_size, shuffle=shuffle)

class MultiTaskDataset():
    def __init__(self, datasets):
        self.tasks = datasets
        self.n_tasks = len(self.tasks)

    def split_shuffle(self, test_ratio=0.2, train_size=None, batch_size=128, shuffle=True, spit=True):
        self.loaders = [task.split_shuffle(test_ratio, train_size, batch_size, shuffle, spit=True) for task in self.tasks]

    def sample_tasks(self,n_tasks=5):
        self.selected_n_tasks = n_tasks
        return random.sample(list(enumerate(self.loaders)),n_tasks)

        
##########################################################################################
# PURGATORY
##########################################################################################

