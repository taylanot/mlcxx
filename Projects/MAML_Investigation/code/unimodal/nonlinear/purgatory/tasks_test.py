import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

class TaskDistribution():
    def __init__(self):
        pass 

    def sample_data(self):
        pass 

    def sample_task(self):
        pass 

class SineTasks(TaskDistribution):
    def __init__(self, dim, x_info=[0,2], a_info=[1,1], p_info=[0,1]):
        super(SineTasks, self).__init__()

        self.a_bound = (a_info[0] - a_info[1], a_info[0] + a_info[1])
        self.p_bound = (p_info[0] - p_info[1], p_info[0] + p_info[1])
        self.a_dist = dist.MultivariateNormal(torch.ones(dim)*a_info[0], \
                torch.eye(dim)*a_info[1])
        self.p_dist = dist.MultivariateNormal(torch.ones(dim)*p_info[0], \
                torch.eye(dim)*p_info[1])
        self.x_dist = dist.MultivariateNormal(torch.ones(dim)*x_info[0], \
                torch.eye(dim)*x_info[1])
        self.dim = dim

    def sample_task(self):
        self.a = self.a_dist.sample()
        self.p = self.p_dist.sample()
        self.task = (self.a, self.p)
        return self.task

    def sample_data(self, task, size, std_y=0):
        x = self.x_dist.sample(torch.Size([size]))
        y = self.f(task,x) + torch.FloatTensor(x.shape[0],self.dim).normal_(0,std_y)
        return x, y

    def f(self, task, x):
        return  torch.sin(x+task[1]) @ task[0].reshape(-1,1)

    def boundaries(self,x):
        assert self.dim == 1
        return [self.f((self.a_bound[i],self.p_bound[i]),x) for i in range(2)]

class LinearTasks(TaskDistribution):
    def __init__(self,dim, x_info=[0,1], m_info=[1,2]):
        super(LinearTasks, self).__init__()

        self.m_bound = (m_info[0] - 2*m_info[1], m_info[0] + 2* m_info[1])

        self.m_dist = dist.MultivariateNormal(torch.ones(dim)*m_info[0], \
                torch.eye(dim)*m_info[1])

        self.x_dist = dist.MultivariateNormal(torch.ones(dim)*x_info[0], \
                torch.eye(dim)*x_info[1])

        self.dim = dim

    def sample_task(self):
        self.m = self.m_dist.sample()
        self.task = self.m
        return self.task

    def sample_data(self, task, size, std_y=0):
        x = self.x_dist.sample(torch.Size([size]))
        y = self.f(task,x) + torch.FloatTensor(x.shape[0],1).normal_(0,std_y)
        print(y)
        return x, y

    def f(self,task, x):
        return x @ task.reshape(-1,1)

    def boundaries(self,x):
        return [self.f(self.m_bounds[i],x) for i in range(2)]


task_dist = LinearTasks(1)
for i in range(10):
    x, y = task_dist.sample_data(task_dist.sample_task(), 1000)
    plt.scatter(x, y)
plt.savefig('plot.pdf')
