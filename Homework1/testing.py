import numpy

coordinate = (3, 2)
positions = numpy.array([
    [1, 1],
    [2, 2],
    [3, 2],
    [4, 4],
    [5, 6],
    [7, 8],
    [3, 2]
], dtype=int)

probs = numpy.zeros(shape=positions.shape[0], dtype=float)
matching_mask = numpy.all(positions == coordinate, axis=1)
probs[matching_mask] = 1.0 / numpy.count_nonzero(matching_mask)
print(probs)