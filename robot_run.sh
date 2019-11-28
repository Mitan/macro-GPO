#!/bin/bash


for loc in $(seq 0 2 34)
    do
        python2  robot.py $loc &
    done
