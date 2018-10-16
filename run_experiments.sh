#!/bin/bash


for loc in $(seq 66 10 316)
    do
        python  main.py $loc 3 &
    done
