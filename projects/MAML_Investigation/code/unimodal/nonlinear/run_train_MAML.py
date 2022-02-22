from train_MAML import ex
from sacred.observers import FileStorageObserver

dims = [1,2,10,50]
NAME = "MAML_Training_noiseless"
ex.observers.append(FileStorageObserver(NAME))

for dim in dims:
    r = ex.run(config_updates={'config.dim': dim}, options={'--name': NAME})


