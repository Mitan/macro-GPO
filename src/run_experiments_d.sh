#!/bin/bash


for loc in  181 182 183 184 185 186 187 188 189 190
    do
        python  deterministic_check_script.py $loc &
    done
