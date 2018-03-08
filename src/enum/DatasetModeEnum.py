def enum(**enums):
    return type('Enum', (), enums)


DatasetModeEnum = enum(Generate=1,
                       Load=2)
