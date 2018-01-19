#!/bin/bash

# args: seed, h
for beta in 0.05 0.1 0.5 1.0 2.0 5.0 10.0
    do
        python  betaTest.py $1 $2 ${beta} &
    done
