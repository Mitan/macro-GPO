#!/bin/bash


for loc in $(seq 0 1 24)
    do
        python2  branin.py $loc &
    done