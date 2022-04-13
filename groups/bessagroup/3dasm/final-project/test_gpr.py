import xarray as xr
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error
from GPy.models import GPRegression

dataset = xr.load_dataset('dataset.sim')
X, y = dataset['Running_Variables'].values, dataset['Results'].values

Xtrn, Xtst, ytrn, ytst = train_test_split(X,y, test_size=0.8)

model = GPRegression(Xtrn, ytrn)
print(model)
model.optimize()
mean, var = model.predict(Xtst)
print(mean)
print(mean_squared_error(ytst,mean))

