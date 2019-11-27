#!/bin/bash


for loc in $(seq 0 20 280)
    do
        python2  main.py $loc &
    done
