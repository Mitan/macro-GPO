#!/bin/bash


for loc in $(seq 0 10 90)
    do
        python2  branin.py $loc &
    done