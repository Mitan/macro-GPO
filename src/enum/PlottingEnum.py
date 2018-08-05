
def enum(**enums):
    return type('Enum', (), enums)


PlottingMethods = enum(TotalReward=1,
                       SimpleRegret=2,
                       Nodes=3,
                       TotalRewardBeta=4,
                       SimpleRegretBeta=5,
                       CumulativeRegret=6)