#!/bin/bash


for loc in $(seq 0 1 35)
    do
        python2  robot.py $loc &
    done
