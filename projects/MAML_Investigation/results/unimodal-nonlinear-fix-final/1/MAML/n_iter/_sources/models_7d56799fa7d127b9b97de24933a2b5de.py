import torch

class Bayes():
    def __init__(self):
        pass

    def fit(self, a):
        self.m_amplitude = a[0]
        self.m_phase= a[1]

    def predict(self,X):
        return   (torch.sin(X+self.m_phase) @ self.m_amplitude.T).reshape(-1,1)

class KernelRidge():
    def __init__(self, lmbda=0, l=1):
        self.lmbda= lmbda
        self.kernel = self.RBF(l)

    def fit(self, D):
        X, Y = D
        self.alpha = torch.inverse(self.kernel(X,X) + self.lmbda*torch.eye(X.shape[0])) @ Y
        self.X = X

    def predict(self, X):
        return (self.kernel(X, self.X)) @ self.alpha

    class RBF():
        def __init__(self,l=1):
            self.l = l

        def pairwise_l2_distance(self, x,y):
            D = -2 * x @  y.T + torch.sum(y**2, axis=1) + torch.sum(x**2, axis=1)[:,None]
            D[D<0] = 0.
            return D

        def __call__(self, x, xp=None):
            if not isinstance(xp, torch.Tensor):
                xp = x
            return torch.exp(-0.5 * self.pairwise_l2_distance(x,xp)/self.l**2)

#class MAML(torch.nn.Module):
#    def __init__(self, in_feature, n_neuron, out_feature, n_hidden=0, activation_tag='tanh'):
#        super(MAML, self).__init__()
#
#        self.layer_in = torch.nn.Linear(in_feature, n_neuron,bias=True)
#        self.layer_out = torch.nn.Linear(n_neuron, out_feature,bias=True)
#        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_neuron, n_neuron, bias=True) for i in range(n_hidden)])
#        self.activations = {'sigmoid':torch.nn.Sigmoid, 'tanh':torch.nn.Tanh, 'softsign':torch.nn.Softsign,'relu':torch.nn.ReLU, 'tanshrink':torch.nn.Tanhshrink}
#        self.activation = self.activations[activation_tag]()
#        self.loss_fn = torch.nn.MSELoss()
#
#    def forward(self,x):
#        x = self.activation(self.layer_in(x))
#        for layer in self.layers:
#            x = self.activation(layer(x))
#        x = self.layer_out(x)
#        return x
#
#    def fit(self, D, lr=0.001, n_iter=1, load=False):
#        x, y = D
#        if load:
#            self.load()
#        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#        for epoch in range(n_iter):
#            optimizer.zero_grad()
#            loss = self.loss_fn(self.forward(x), y)
#            loss.backward()
#            optimizer.step()
#
#    def predict(self, x):
#        return self.forward(x)
#
#    @torch.no_grad()
#    def test(self, D):
#        x, y = D
#        return self.loss_fn(self.forward(x), y)
#
#    def load(self, path='model.pt'):
#        self.load_state_dict(torch.load(path))
