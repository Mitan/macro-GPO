#!/bin/bash


for loc in $(seq 0 1 30)
    do
        python2  branin.py $loc &
    done