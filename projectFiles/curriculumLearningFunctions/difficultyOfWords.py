def numberOfComplexWordInInput(_1, _2, inputIndices, _3):
    return len(list(filter(lambda x: x > 5000, inputIndices)))


def numberOfComplexWordInInputOrOutput(_1, _2, inputIndices, outputIndices):
    return len(list(filter(lambda x: x > 5000, inputIndices))) + len(list(filter(lambda x: x < 5000, outputIndices)))
