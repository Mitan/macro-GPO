#!/bin/bash

for f in 8
do
    for beta in  0
    do
        for loc in 0 1 2 3 4 5 6 7 8 9
        do
        python  main_testing_script.py $f $beta $loc &
        done
    done
done
