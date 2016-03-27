#!/bin/bash

for beta in  0 1
do
    for f in 1 6 7
    do
        for loc in 0 1 2 3 4
        do
        python  main_testing_script.py $f $beta $loc &
        done
    done
done

for beta in   1
do
    for f in 3
    do
        for loc in 0 1 2 3 4
        do
        python  main_testing_script.py $f $beta $loc &
        done
    done
done

for beta in 0
do
    for f in 4
    do
        for loc in 0 1 2 3 4
        do
        python  main_testing_script.py $f $beta $loc &
        done
    done
done