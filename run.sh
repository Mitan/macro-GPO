#!/usr/bin/env bash

for start in $(seq 0 5 50)
        do
           python2  main.py ${start} &
        done