import torch
import torch.distributions as dist 

class TaskDistribution():
    def __init__(self):
        pass 
    
    def sample_data(self):
        pass 

    def sample_task(self):
        pass 

class LinearTasks(TaskDistribution):
    def __init__(self,dim=1, std_y=0., x_info=[0,2], a_info=[4,1]):
        super(LinearTasks, self).__init__()

        self.a_bound = (a_info[0] - 2*a_info[1], a_info[0] + 2*a_info[1])
        self.x_bound = (x_info[0] - 2*x_info[1], x_info[0] + 2*x_info[1])

        self.a_dist = dist.MultivariateNormal(torch.ones(dim)*a_info[0], \
                torch.eye(dim)*a_info[1])

        self.x_dist = dist.MultivariateNormal(torch.ones(dim)*x_info[0], \
                torch.eye(dim)*x_info[1])

        self.dim = dim
        self.std_y = std_y

    def sample_task(self):
        self.a = self.a_dist.sample()
        self.task = self.a
        return self.task

    def sample_data(self, task, size, std_y=0):
        x = self.x_dist.sample(torch.Size([size]))
        y = self.f(task,x) + torch.FloatTensor(x.shape[0],1).\
                normal_(0,self.std_y)
        return x, y

    def f(self,task, x):
        return x @ task.reshape(-1,1)

    def boundaries(self,x):
        return [self.f(torch.Tensor([self.a_bound[i]]),x) for i in range(2)]

class SineTasks(TaskDistribution):
    def __init__(self, dim=1, std_y =0, x_info=[0,2], a_info=[1,1],\
            p_info=[0,1]):

        super(SineTasks, self).__init__()

        self.a_bound = (a_info[0] - a_info[1], a_info[0] + a_info[1])
        self.p_bound = (p_info[0] - p_info[1], p_info[0] + p_info[1])
        self.x_bound = (x_info[0] - 2*x_info[1], x_info[0] + 2*x_info[1])

        self.a_dist = dist.MultivariateNormal(torch.ones(dim)*a_info[0], \
                torch.eye(dim)*a_info[1])
        self.p_dist = dist.MultivariateNormal(torch.ones(dim)*p_info[0], \
                torch.eye(dim)*p_info[1])
        self.x_dist = dist.MultivariateNormal(torch.ones(dim)*x_info[0], \
                torch.eye(dim)*x_info[1])
        self.dim = dim
        self.std_y= std_y

    def sample_task(self):
        self.a = self.a_dist.sample()
        self.p = self.p_dist.sample()
        self.task = (self.a, self.p)
        return self.task

    def sample_data(self, task, size):
        x = self.x_dist.sample(torch.Size([size]))
        y = self.f(task,x) + torch.FloatTensor(x.shape[0],1).\
                normal_(0,self.std_y)
        return x, y

    def f(self, task, x):
        return  torch.sin(x+task[1]) @ task[0].reshape(-1,1)

    def boundaries(self,x):
        return [self.f((torch.Tensor([self.a_bound[i]]),\
                torch.Tensor([self.p_bound[i]])),x) for i in range(2)]


