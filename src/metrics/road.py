from RegretCalculator import *
from RoadAverager import *
"""
GetRoadBeta3Rewards()
print
GetRoadBeta2Rewards()
GetRoadBeta2Regrets()
GetRoadBeta3Regrets()
GetRoadBeta3Rewards()
print
GetRoadBeta2Rewards()
print

GetRoadTotalRewards()
GetRoadTotalRegrets()

GetRoad_H2Full_TotalRewards()
print
GetRoadTotalRegrets_H2Full()

GetRoadTotalRewards()
"""
# GetRoadBeta3Rewards()

# GetRoadBeta2Rewards()
# GetRoadBeta2Regrets()
# GetRoadBeta3Regrets()
seeds = list(set(range(43)) - set([21, 41, 10, 14, 18, 22, 26, 33]))
seeds = range(43)
GetRoadTotalRewards(seeds)
GetRoadTotalRegrets(seeds)
