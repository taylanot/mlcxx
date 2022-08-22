#!/usr/bin/env bash

for dim in 1 
do
  for Ntrn in 1
do
    python3 EE_fix.py with model_tag="MAML" run_tag="std_y" config.Ntrn=$Ntrn config.dim=$dim  & 
    python3 EE_fix.py with model_tag="GD" run_tag="std_y" config.Ntrn=$Ntrn config.dim=$dim  & 
done
  wait
done


