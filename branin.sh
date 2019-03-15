#!/bin/bash


for loc in $(seq 0 2 60)
    do
        python2  branin.py $loc &
    done