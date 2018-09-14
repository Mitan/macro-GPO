#!/bin/bash


for loc in $(seq 66 25 316)
    do
        python  main.py $loc &
    done
