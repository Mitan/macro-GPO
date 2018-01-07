#!/bin/bash


for loc in 0 3 6 9 12 15 18 21 24 27 30 33
    do
        python  h4_script.py $loc &
    done
