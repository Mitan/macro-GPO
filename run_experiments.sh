#!/bin/bash


for loc in $(seq 0 50 300)
    do
        python2  main.py $loc &
    done
