#!/bin/bash

# args: seed, h
for loc in 0 3 6 9 12 15 18 21 24 27 30 33
    do
        python  betaTest.py ${loc} 3 1  &
    done
