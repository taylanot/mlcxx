#!/bin/bash

python3 run_EE.py --model_tag=KernelRidge --run_tag=std_y

python3 run_EE.py --model_tag=KernelRidge --run_tag=Ntrn

python3 run_EE.py --model_tag=KernelRidge --run_tag=c_phase

python3 run_EE.py --model_tag=KernelRidge --run_tag=c_amplitude

python3 run_EE.py --model_tag=KernelRidge --run_tag=n_iter