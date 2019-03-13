#!/bin/bash


for loc in $(seq 0 2 100)
    do
        python2  branin.py $loc &
    done