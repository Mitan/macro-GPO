def enum(**enums):
    return type('Enum', (), enums)


PlottingEnum = enum(AverageTotalReward=1,
                    SimpleRegret=2,
                    AverageRewardBeta=3)
