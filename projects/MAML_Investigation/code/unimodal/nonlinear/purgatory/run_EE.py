from EE import ex
from sacred.observers import FileStorageObserver
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', '-m', help="model options", type= str,\
        default=None)
parser.add_argument('--run_tag', '-r', help="run options", type= str,\
        default=None)

# usage: test_args_4.py [-h] [--foo FOO] [--bar BAR]
# 
# optional arguments:
#   -h, --help         show this help message and exit
#   --model_tag, -m assign a model_tag

args = parser.parse_args()


NAME = "unimodal-nonlinear" 
dims = [1]#,2,10,50]
Ntrns = [1]#,2,10,50]

ex.observers.append(FileStorageObserver.create(os.path.join(NAME,\
                                    str(args.model_tag),str(args.run_tag))))
#ids = []
#ids.append(({'id':,id_,'dim':dim,'Ntrn':Ntrn}))
#id_ = 1
if args.run_tag != 'Ntrn':
    for dim in dims:
        for  Ntrn in Ntrns:
            res = ex.run(config_updates={'config.dim': dim, 'config.Ntrn': Ntrn, \
            'model_tag':args.model_tag, 'run_tag':args.run_tag})
            #id_ +=1 

elif args.run_tag == 'Ntrn':
    for dim in dims:
        res = ex.run(config_updates={'config.dim': dim, \
        'model_tag':args.model_tag, 'run_tag':args.run_tag})
 


