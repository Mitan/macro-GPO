def enum(**enums):
    return type('Enum', (), enums)


PlottingEnum = enum(AverageTotalReward=1,
                    SimpleRegret=2,
                    AverageRewardBeta=3,
                    AverageRewardFull=4,
                    SimpleRegretFull=5,
                    SimpleRegretRollout=6,
                    AverageTotalRewardRollout=7)
