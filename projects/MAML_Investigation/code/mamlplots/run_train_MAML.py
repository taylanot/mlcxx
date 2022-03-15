from train_multitask import ex
from sacred.observers import FileStorageObserver

dims = [1]
NAME = "multitask_plots"
ex.observers.append(FileStorageObserver(NAME))

for dim in dims:
    r = ex.run(config_updates={'config.dim': dim},  options={'--name': NAME})


