#!/bin/bash


for loc in {66 .. 101}
    do
        python  main.py $loc &
    done
