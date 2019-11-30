#!/bin/bash


for loc in $(seq 66 1 101)
    do
        python2  simulated.py $loc &
    done
