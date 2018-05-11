#!/bin/bash

for i in {66..101}
	do
		# mkdir ./temp/seed$i
		# cp -r ./road_tests/new_new_new_beta2/seed"$i"/"$1" ./temp/seed$i
		# cp -r ./noise_robot_tests/release/all_tests_release/seed"$i"/bbo-llp ./releaseTests/updated_release/robot/all_tests_release/seed$i
		# cp -r ./releaseTests/updated_release//all_tests_release/seed"$i"/bbo-llp ./releaseTests/updated_release/robot/all_tests_release/seed$i
		mv  ./sim/rewards-sAD/seed"$i"/gp-bucb ./sim/rewards-sAD/seed"$i"/bucb
		# echo $i
	done

