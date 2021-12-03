#!/usr/bin/env bash

python3 numba_EE.py with "config.model='Ridge'" "run_tag="$1
python3 numba_EE.py with "config.model='SGD'" "run_tag="$1

#python3 numba_EE.py with "config.model='Bayes'" "run_tag="$1
#python3 numba_EE.py with "config.model='Linear'" "run_tag="$1

