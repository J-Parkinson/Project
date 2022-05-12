from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.embeddingType import embeddingType


def getMaxLens(dataset, embedding, restrict=2000000):
    if dataset == datasetToLoad.wikilarge:
        if embedding == embeddingType.bert:
            length = 256
        else:
            length = 82
    elif dataset == datasetToLoad.wikismall:
        if embedding == embeddingType.bert:
            length = 232
        else:
            length = 82
    else:
        if embedding == embeddingType.bert:
            length = 86
        else:
            length = 67
    return min(length, restrict)
