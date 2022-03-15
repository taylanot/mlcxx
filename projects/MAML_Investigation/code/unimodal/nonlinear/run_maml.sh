#!/usr/bin/env bash

for dim in 1 2 10 50 
do
  for Ntrn in 1 2 10 50
do
    python3 EE.py with model_tag="MAML" run_tag="std_y" config.Ntrn=$Ntrn config.dim=$dim  & 
done
  wait
done

for dim in 1 2 10 50 
do
  for Ntrn in 1 2 10 50
do
    python3 EE.py with model_tag="MAML" run_tag="n_iter" config.Ntrn=$Ntrn config.dim=$dim  & 
done
  wait
done


for dim in 1 2 10 50 
do
  for Ntrn in 1 2 10 50
do
    python3 EE.py with model_tag="MAML" run_tag="Ntrn"  config.dim=$dim  & 
done
  wait
done

#for dim in 1 2 10 50 
#do
#  for Ntrn in 1 2 10 50
#do
#    python3 EE.py with model_tag="MAML" run_tag="c_phase" config.Ntrn=$Ntrn config.dim=$dim  & 
#done
#  wait
#done
#
#for dim in 1 2 10 50 
#do
#  for Ntrn in 1 2 10 50
#do
#    python3 EE.py with model_tag="MAML" run_tag="c_amplitude" config.Ntrn=$Ntrn config.dim=$dim  & 
#done
#  wait
#done


