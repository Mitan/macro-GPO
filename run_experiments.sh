#!/bin/bash


for loc in $(seq 175 10 325)
    do
        python  main.py $loc &
    done
