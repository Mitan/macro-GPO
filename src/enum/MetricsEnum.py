
def enum(**enums):
    return type('Enum', (), enums)


MetricsEnum = enum(AverageTotalReward=1,
                   SimpleRegret=2)