#!/bin/bash


for loc in $(seq 0 1 10)
    do
        python2  robot.py $loc &
    done
