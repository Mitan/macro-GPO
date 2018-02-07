from SimulatedAverager import *
from RegretCalculator import *

"""
GetSimulatedBeta2Regrets()
GetSimulatedBeta2Rewards()


GetSimulatedBeta3Regrets()
GetSimulatedBeta3Rewards()

GetSimulatedTotalRewards_our()
GetSimulatedTotalRegrets_our()
for ei in [True, False]:
    GetSimulatedTotalRewards(ei)
    GetSimulatedTotalRewards_onlyH4(ei)

    GetSimulatedTotalRegrets(ei)
    GetSimulatedTotalRegrets_onlyH4(ei)
    print
"""

# GetSimulatedBeta2Rewards()

print
# GetSimulatedBeta3Rewards()

for ei in [True]:
    GetSimulatedTotalRewards(ei)
    print
    GetSimulatedTotalRegrets(ei)
