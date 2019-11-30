#!/bin/bash


for loc in $(seq 334 25 380)
    do
        python2  main.py $loc &
    done
