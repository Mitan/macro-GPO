#!/bin/bash


for loc in $(seq 0 4 96)
    do
        python2  branin.py $loc &
    done