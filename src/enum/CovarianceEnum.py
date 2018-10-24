def enum(**enums):
    return type('Enum', (), enums)


CovarianceEnum = enum(SquareExponential=1)

