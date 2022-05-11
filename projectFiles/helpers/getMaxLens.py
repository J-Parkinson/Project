from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.embeddingType import embeddingType


def getMaxLens(dataset, embedding, restrict=2000000):
    if dataset == datasetToLoad.wikilarge:
        if embedding == embeddingType.bert:
            length = 254
        else:
            length = 82
    elif dataset == datasetToLoad.wikismall:
        if embedding == embeddingType.bert:
            length = 230
        else:
            length = 82
    else:
        if embedding == embeddingType.bert:
            length = 84
        else:
            length = 67
    return min(length, restrict)
