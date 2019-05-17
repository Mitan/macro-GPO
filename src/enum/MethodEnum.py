def enum(**enums):
    return type('Enum', (), enums)


Methods = enum(Exact=1,
               Anytime=2)
