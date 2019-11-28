#!/bin/bash


for loc in $(seq 66 7 94)
    do
        python2  simulated.py $loc &
    done
