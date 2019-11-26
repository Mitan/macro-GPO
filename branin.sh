#!/bin/bash


for loc in $(seq 40 2 100)
    do
        python2  branin.py $loc &
    done