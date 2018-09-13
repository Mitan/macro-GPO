
def enum(**enums):
    return type('Enum', (), enums)


MetricsEnum = enum(TotalReward=1,
                   SimpleRegret=2)