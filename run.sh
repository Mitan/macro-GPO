#!/usr/bin/env bash

for start in $(seq 66 10 171)
        do
           python2  main.py ${start} &
        done