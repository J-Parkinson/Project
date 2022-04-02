def numberOfSimpleWordInInput(_1, _2, inputIndices, _3):
    return len(list(filter(lambda x: x < 5, inputIndices)))


def numberOfSimpleWordInInputOrOutput(_1, _2, inputIndices, outputIndices):
    return len(list(filter(lambda x: x < 5, inputIndices))) + len(list(filter(lambda x: x < 5, outputIndices)))
