#!/usr/bin/env bash

for dim in 1 2 10 50 
do
    echo Starting Block-1:Ntrn
    python3 numba_EE.py with "model_tag='Bayes'" "run_tag='Ntrn'" "config.dim="$dim   
    wait
  for Ntrn in 1 2 10 50
do
    echo Starting Block-2:std_y
    python3 numba_EE.py with "model_tag='Bayes'" "run_tag='std_y'" "config.Ntrn="$Ntrn "config.dim="$dim   

    echo Starting Block-3:m
    python3 numba_EE.py with "model_tag='Bayes'" "run_tag='m'" "config.Ntrn="$Ntrn "config.dim="$dim   

    echo Starting Block-4:c
    python3 numba_EE.py with "model_tag='Bayes'" "run_tag='c'" "config.Ntrn="$Ntrn "config.dim="$dim   

    echo Starting Block-5:b
    python3 numba_EE.py with "model_tag='Bayes'" "run_tag='b'" "config.Ntrn="$Ntrn "config.dim="$dim   

done
done

wait 
for Ntrn in 1 2 10 50
do
  echo Starting Block-7:dim
  python3 numba_EE.py with "model_tag='Bayes'" "run_tag='dim'" "config.Ntrn="$Ntrn
done

