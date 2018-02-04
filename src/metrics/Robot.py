from RobotAverager import *
from RegretCalculator import *

# GetRobotTotalRegrets_beta2()
print
# GetRobotTotalRegrets_beta3()
print
# GetRobotBeta2Rewards()
print
# GetRobotBeta3Rewards()
"""
GetRobotTotalRewards()
GetRobotTotalRewards_onlyH4()
# GetRobotTotalRewards_ours()
"""
# GetRobotTotalRegrets()
# GetRobotTotalRegrets_onlyH4()
# GetRobotTotalRegrets_ours()

GetRobotBeta2Rewards()
GetRobotBeta3Rewards()

GetRobot_H2Full_TotalRewards()
GetRobotTotalRegrets_H2Full()

for ei in [True, False]:
    GetRobotTotalRewards(ei)
    GetRobotTotalRegrets(ei)

    # GetRobotTotalRewards_onlyH4(ei)
    # GetRobotTotalRegrets_onlyH4(ei)