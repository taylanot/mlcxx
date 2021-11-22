import pickle
import os 
import torch
import pandas

def read_pickle(filename):

    """ Read Pickle files """

    name, ext = os.path.splitext(filename)
    assert ext == '.pkl' or ext == '.pickle', 'File extension should be .pkl or .pickle'
    loading = open(filename,"rb")
    return  pickle.load(loading)

def tensorize_DataFrame(frame):

    """ Convert pandas DataFrame to pytorch Tensor object """

    assert isinstance(frame,pandas.DataFrame), 'You have to provide pandas.DataFrame.'
    return torch.Tensor([frame[frame.keys()[i]] for i in range(len(frame.keys()))])

class ewma():

    """ Exponential Weighted Moving Average """

    def __init__(self, rho=0.95):
        self.rho = rho
        self.s_prev = 0.
        self.counter = 1. 
        self.avg = []
        self.avg_bc = []

    def __call__(self,s):

        self.s_cur = self.s_prev * self.rho + s * (1 - self.rho)
        self.s_cur_bc = self.s_cur / (1 - self.rho**(self.counter))

        self.s_prev = self.s_cur
        
        self.avg.append(self.s_cur)
        self.avg_bc.append(self.s_cur_bc)

        self.counter += 1. 

    def get_avg(self,bias_correction=False):
        if bias_correction:
            return self.avg_bc
        else:
            return self.avg

class network_tools():

    """ Network tools """

    def __init__(self):
        super(network_tools,self).__init__()

        self.activation_values = {}
        self.init_linear_weight = torch.nn.init.xavier_uniform_
        self.init_linear_bias = torch.nn.init.zeros_

    def _grad_norm(self):
        self.total_norm = 0.
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            param_norm = p.grad.norm(2)
            self.total_norm += param_norm.item() ** 2
        self.total_norm = self.total_norm ** (1. / 2)
        return self.total_norm

    def _weight_init_linear(self,m):
        if isinstance(m, torch.nn.Linear):
            self.init_linear_weight(m.weight)

    def _bias_init_linear(self,m):
        if isinstance(m, torch.nn.Linear):
            self.init_linear_bias(m.bias)
    
    def _load_parameters(self,model_path):
        self.load_model_path = model_path
        self.load_state_dict(torch.load(model_path))

    
    def _register_activations(self, name):                  
        def hook(model, input, output):
            self.activation_values[name] = self.activate(output.detach())
        return hook                              

    def _show_parameters(self):
        for name, v in self.state_dict().items():
            param = torch.nn.Parameter(v)
            print("Parameters from", name)
            print(param)

    def _reset_linear_layers(self):
        self.apply(self._weight_init_linear)
        self.apply(self._bias_init_linear)

    def _reset_load_parameters(self,model_path=None):
        if model_path == None:
            self._load_parameters(self.load_model_path)
        else:
            self._load_parameters(self.model_path)

    def list_parameters(self):
        return list(self.parameters())
            
class RBF(torch.nn.Module):
    def __init__(self, in_features, out_features, domain=[-5,5], sigma=0.1, functional_key='gaussian'):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.centers = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.centers = torch.linspace(domain[0],domain[1],out_features*in_features).view(out_features, in_features)
        #self.log_sigmas = torch.nn.Parameter(torch.Tensor(out_features))
        self.log_sigmas = torch.ones((out_features))*sigma
        self.basis_func = getattr(self,functional_key)
        #self.reset_parameters()

    def gaussian(self, alpha):
        phi = torch.exp(-1*alpha.pow(2))
        return phi       

    def matern32(self, alpha):
        phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
        return phi

    def quadratic(self, alpha):
        phi = alpha.pow(2)
        return phi

    def linear(self, alpha):
        phi = alpha
        return phi

    def inverse_quadratic(self, alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        return phi

    def multiquadric(self, alpha):
        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def inverse_multiquadric(self, alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def spline(self, alpha):
        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
        return phi

    def poisson_one(self, alpha):
        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        return phi

    def poisson_two(self, alpha):
        phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
        return phi

    def matern32(self, alpha):
        phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
        return phi

    def matern52(self, alpha):
        phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
        * alpha.pow(2))*torch.exp(-5**0.5*alpha)
        return phi

    def reset_parameters(self):
        #torch.nn.init.normal_(self.centres, 0, 1)
        #torch.nn.init.constant_(self.log_sigmas, 0)
        pass

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        #print(distances.size())
        #print(torch.exp(self.log_sigmas).unsqueeze(0).size())

        return self.basis_func(distances)

def grad(f,x,t):
  gradients = []
  if t > 1:
    retain_graph = True
    create_graph = True
  else:
    retain_graph = True
    create_graph = True
  for order in range(1,t+1):
    print(order)
    if order == 1:
      f = f.unsqueeze(0)
    else:
      f = gradients[-1]
    dx = torch.zeros_like(f)
    gradient = []
    for i in range(dx.shape[0]):
      for j in range(dx.shape[-1]):
        dx[i,:,j]=1.
        df,= torch.autograd.grad(f,x,dx,retain_graph=retain_graph,create_graph=create_graph, allow_unused=True)
        dx[i,:,j]=0.
        gradient.append(df)
    if all(df is None for df in gradient):
      gradients.append(None)
    else:
      gradients.append(torch.stack(gradient))
  return gradients


