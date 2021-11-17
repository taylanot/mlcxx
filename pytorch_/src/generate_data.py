import numpy as np
import random
from math import ceil, log, floor 
from .dataset import * 

class LinearTasks():
    def __init__(self, m_bounds=[1,2],b_bounds=None, diff='easy'):
        self.m_bounds = m_bounds
        self.b_bounds = b_bounds
        if self.b_bounds is None:
            self.b_bounds = [0,0]
        self.boundary_tasks = [(self.m_bounds[i],self.b_bounds[i]) for i in range(2)]
        self.diff = diff

    def sample_task(self):
        self.m = torch.FloatTensor(1,1).uniform_(self.m_bounds[0], self.m_bounds[1])
        if self.diff is 'easy':
            self.b = torch.zeros(1)
        else:
            assert self.b_bounds is not None, "Provide bounds for the intercept!"
            self.b = torch.FloatTensor(1,1).uniform_(self.b_bounds[0], self.b_bounds[1])
        self.task = (self.m,self.b)
        return self.task

    def sample_data(self, task, domain_bounds, size, noise=False):
        x = torch.FloatTensor(size,1).uniform_(domain_bounds[0], domain_bounds[1])
        if noise:
            y = self.f(task,x) + torch.FloatTensor(x.size()).normal_()
        else:
            y = self.f(task,x)
        return x, y

    def f(self,task, x):
        return task[0] * x + task[1]

    def boundaries(self,x):
        return [self.f((self.m_bounds[i],self.b_bounds[i]),x) for i in range(2)]

class SineTasks():
    def __init__(self, a_bounds=[1,2], p_bounds=None, diff='easy'):
        self.a_bounds = a_bounds
        self.p_bounds = p_bounds
        if self.p_bounds is None:
            self.p_bounds = [0,0]
        self.boundary_tasks = [(self.a_bounds[i],self.p_bounds[i]) for i in range(2)]
        self.diff = diff

    def sample_task(self):
        if self.diff is 'easy':
            self.a = torch.FloatTensor(1,1).uniform_(self.a_bounds[0], self.a_bounds[1])
            self.p = torch.zeros(1)
        elif self.diff is 'hard':
            assert self.p_bounds is not None, "Provide bounds for the phase!"
            self.a = torch.FloatTensor(1,1).uniform_(self.a_bounds[0], self.a_bounds[1])
            self.p = torch.FloatTensor(1,1).uniform_(self.p_bounds[0], self.p_bounds[1])
        elif self.diff is 'medium':
            self.a = torch.ones(1)
            self.p = torch.FloatTensor(1,1).uniform_(self.p_bounds[0], self.p_bounds[1])
        self.task = (self.a, self.p)
        return self.task

    def sample_data(self, task, domain_bounds, size, noise=False):
        x = torch.FloatTensor(size,1).uniform_(domain_bounds[0], domain_bounds[1])
        if noise:
            y = self.f(task, x) + torch.FloatTensor(x.size()).normal_()
        else:
            y = self.f(task, x)
        return x,y

    def f(self, task, x):
        return task[0] * torch.sin(x + task[1]) 

    def boundaries(self,x):
        return [self.f((self.a_bounds[i],self.p_bounds[i]),x) for i in range(2)]


class TaskDistribution():
    def __init__(self, task_def, domain_bounds):
        self.task_def = task_def
        self.domain_bounds = domain_bounds
    
    def sample_data(self, task, size, noise=False):
        #self.sample_task()
        return self.task_def.sample_data(task, self.domain_bounds, size, noise=noise)

    def sample_task(self):
        self.task = self.task_def.sample_task()
        return self.task


class FunctionalDatasetGenerator():

    """ Dataset generator from given functional for meta-learning """

    def __init__(self, n_tasks=5, set_size=1000, **kwargs):

        self.amplitude_bounds = kwargs.get('amplitude_bounds', [0.1, 1])
        self.phase_bounds = kwargs.get('phase_bounds', [0, 0])
        self.domain_bounds = kwargs.get('domain_bounds', [-5.0, 5.0])
        self.dim = kwargs.get('dim',2)
        self.function_family = kwargs.get('functions',[torch.sin, torch.cos])
        self.task_complexity = kwargs.get('complexity', 'easy')

        self.set_size = set_size
        self.n_tasks = n_tasks

        self.kwargs = kwargs 

    def create(self, spit=False, keep_features=True, **kwargs):
        self.tasks = self.sample_easy_task(self.kwargs.get('sample_task','grid'))
        self.features = self.sample_features(self.kwargs.get('sample_feature','grid'))
        #print(self.features)
        #kwargs = {'scale_input':'standard', 'scale_output':'standard'}
        if keep_features:
            self.datasets = [SingleTaskDataset(torch.utils.data.TensorDataset(self.features[0],self.tasks[0][i] * self.function_family[self.tasks[2][i]](self.features[0] - self.tasks[1][i])), filter_nan=False,**kwargs) for i in range(self.n_tasks)]
            #print(self.features[0],self.tasks[0][0] * self.function_family[self.tasks[2][0]](self.features[0] - self.tasks[1][0]))
        else:
            self.datasets = [SingleTaskDataset(torch.utils.data.TensorDataset(self.features[i],self.tasks[0][i] * self.function_family[self.tasks[2][i]](self.features[i] - self.tasks[1][i])), filter_nan=False,**kwargs) for i in range(self.n_tasks)]

        if spit:
            return self.datasets

    def resample(self, **kwargs):
        self.create()

    def sample_features(self, method='grid', keep_features=True):
        if method == 'grid' and keep_features==True:
            self.set_size = self.set_size if self.set_size <= 2.**self.dim else pow(2, ceil(log(self.set_size)/log(2)))
            space = torch.linspace(self.domain_bounds[0], self.domain_bounds[1], round(self.set_size**(1. /self.dim)))
            meshgrid = torch.meshgrid([space for i in range(self.dim)])
            meshgrid = [meshgrid[i].reshape(-1,1) for i in range(len(meshgrid))]
            return [torch.cat(meshgrid,1)] * self.n_tasks
        elif method == 'uniform' and keep_features==True:
            samples = torch.FloatTensor(self.set_size,self.dim).uniform_(self.domain_bounds[0],self.domain_bounds[1])
            return [samples for i in range(self.n_tasks)]
        elif method == 'uniform' and keep_features==False:
            return torch.FloatTensor(self.n_tasks,self.set_size,self.dim).uniform_(self.domain_bounds[0],self.domain_bounds[1])

    def sample_hard_task(self, method='grid'):
        if method == 'grid':
            self.n_tasks = self.n_tasks if self.n_tasks == 2.**3. else int(2.**3.)
            n_tasks = round(self.n_tasks**(1. /3.))
            amplitude  = torch.linspace(self.amplitude_bounds[0], self.amplitude_bounds[1], n_tasks)
            phase = torch.linspace(self.phase_bounds[0], self.phase_bounds[1], n_tasks)
            func_index = torch.randint(0,len(self.function_family),(n_tasks,)) 
            meshgrid = torch.meshgrid([amplitude, phase, func_index.type(torch.FloatTensor)])
            meshgrid = [meshgrid[i].reshape(-1,1) for i in range(len(meshgrid))]
            tasks = (meshgrid[0], meshgrid[1], meshgrid[2].type(torch.long))
            return tasks
        elif method == 'uniform':
            amp = torch.FloatTensor(self.n_tasks,1).uniform_(self.amplitude_bounds[0],self.amplitude_bounds[1])
            phs = torch.FloatTensor(self.n_tasks,1).uniform_(self.phase_bounds[0],self.phase_bounds[1])
            func_index = torch.randint(0,len(self.function_family),(self.n_tasks,1)) 
            tasks = (amp,phs,func_index)
            return tasks 
    def sample_medium_task(self, method='grid'):
        if method == 'grid':
            self.n_tasks = self.n_tasks if self.n_tasks == 2.**3. else int(2.**3.)
            n_tasks = round(self.n_tasks**(1. /3.))
            amplitude  = torch.linspace(self.amplitude_bounds[0], self.amplitude_bounds[1], n_tasks)
            phase = torch.linspace(self.phase_bounds[0], self.phase_bounds[1], n_tasks)
            func_index = torch.zeros(phase.size()).int()
            meshgrid = torch.meshgrid([amplitude, phase, func_index.type(torch.FloatTensor)])
            meshgrid = [meshgrid[i].reshape(-1,1) for i in range(len(meshgrid))]
            tasks = (meshgrid[0], meshgrid[1], meshgrid[2].type(torch.long))
            return tasks
        elif method == 'uniform':
            amp = torch.FloatTensor(self.n_tasks,1).uniform_(self.amplitude_bounds[0],self.amplitude_bounds[1])
            phs = torch.FloatTensor(self.n_tasks,1).uniform_(self.phase_bounds[0],self.phase_bounds[1])
            func_index = torch.zeros(phs.size()).int()
            tasks = (amp,phs,func_index)
            return tasks 
    def sample_easy_task(self, method='grid'):
        if method == 'grid':
            self.n_tasks = self.n_tasks if self.n_tasks == 2.**3. else int(2.**3.)
            n_tasks = round(self.n_tasks**(1. /3.))
            amplitude  = torch.linspace(self.amplitude_bounds[0], self.amplitude_bounds[1], n_tasks)
            phase = torch.zeros(amplitude.size())
            func_index = torch.zeros(phase.size()).int()
            meshgrid = torch.meshgrid([amplitude, phase, func_index.type(torch.FloatTensor)])
            meshgrid = [meshgrid[i].reshape(-1,1) for i in range(len(meshgrid))]
            tasks = (meshgrid[0], meshgrid[1], meshgrid[2].type(torch.long))
            return tasks
        elif method == 'uniform':
            amp = torch.FloatTensor(self.n_tasks,1).uniform_(self.amplitude_bounds[0],self.amplitude_bounds[1])
            phs = torch.zeros(amp.size())
            func_index = torch.zeros(phs.size()).int()
            tasks = (amp,phs,func_index)
            return tasks 








