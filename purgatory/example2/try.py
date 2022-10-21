import torch
import time
class Net(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


model = Net(1,1)
x = torch.ones(1)
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(x)

        # get loss for the predicted output
        loss = criterion(outputs, torch.ones(1)*2)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

start = time.time()
for i in range(1000):
    train()
end = time.time()
print(end-start)
