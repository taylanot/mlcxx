import numpy as np
import matplotlib.pyplot as plt
import GPy.models
np.random.seed(24)
################################################################################
def funcy(x, noise = True): # Create Data
  if noise:
    return np.sin(x) + np.random.normal(0,0.1,(x.shape[0],1))
  else:
    return np.sin(x)
################################################################################


X = np.random.normal(0,5,(20,1))                      # Training Data Feature
#X = np.linspace(-1,1,20).reshape(-1,1)
Y = funcy(X)                                          # Training Data Labels

# Initialize model
model = GPy.models.GPRegression(X,Y)                  # Train the model                                                                
print(model)                                          # Print Model Information 

model.optimize_restarts(10)
print(model)                                          # Print model information 
model.plot()           
