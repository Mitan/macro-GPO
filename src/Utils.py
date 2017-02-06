def TupleToLine(tuple_location, dim_1, dim_2):
    float_line =  tuple_location[0] * dim_2 + tuple_location[1] + 1
    return int(float_line)

def LineToTuple(line_location, dim_1, dim_2):
    return (  float((line_location - 1) / dim_2),   float((line_location - 1) % dim_2)  )


