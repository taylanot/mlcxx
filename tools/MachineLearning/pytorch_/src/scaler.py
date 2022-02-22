import torch

class Scaler():
    def __init__(self,x,method='standard',order=None,dim=0,eps=1e-12):
        assert isinstance(x,torch.Tensor), " input should be a torch.Tensor!"
        
        ################################
        # Initialize class attributes
        ################################
        self.x = x
        self._eps = eps
        self._method = method
        self._order = order
        self._dim = dim 

        ################################
        # Calculate statistics on dim
        ################################
        self.mean = self.x.mean(dim= self._dim,keepdim=True)
        self.max = self.x.max(dim= self._dim,keepdim=True)
        self.min = self.x.min(dim= self._dim,keepdim=True)
        self.std = self.x.std(dim= self._dim,keepdim=True)

        ################################
        # Get the right transformation
        ################################
        self.trans = getattr(self,self._method)

    def standard(self,x,mode):

        """ Method: Standard Standardization """

        if mode == 'transform':
            return (x - self.mean) / (self.std + self._eps)
        elif mode == 'retransform':
            return x * (self.std + self._eps) + self.mean

    def minmax(self,x,mode):

        """ Method: Min-Max Standardization """

        if mode == 'transform':
            return (x - self.min.values) / (self.max.values - self.min.values)
        elif mode == 'retransform':
            return x * (self.max.values - self.min.values) + self.min.values

    def norm(self,x,mode):

        """ Method: Normalization """

        ################################
        # PyTorch norm calculator
        ################################
        self.norm = torch.linalg.norm(self.x,dim=self.dim,ord=self._order)

        if mode == 'transform':
            return x / self.norm
        elif mode == 'retransform':
            return x * self.norm

    def __call__(self,x,mode='transform'):
        return self.trans(x, mode)

    #def crop_merge(self,ix):

    #    print(self.std)
    #    [print(getattr(self, a)[:,1:3]) for a in dir(self) if not a.startswith('__') and not a.startswith('_') and not callable(getattr(self, a))]
    #    
    #    #if inx != None and outx != None:
    #    #        self.in_size, self.out_size = len(inx), len(outx)
    #    #        self.input = self.input[:,inx]
    #    #        self.output = self.output[:,outx]
    #    #        self.data = TensorDataset(self.input, self.output)

