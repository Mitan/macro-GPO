def enum(**enums):
    return type('Enum', (), enums)


DatasetEnum = enum(Simulated=1,
                   Road=2,
                   Robot=3)

DatasetModeEnum = enum(Generate=1,
                       Load=2)
