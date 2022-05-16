from projectFiles.helpers.DatasetToLoad import datasetToLoad


def getMaxLens(dataset, embedding, restrict=2000000):
    if dataset == datasetToLoad.wikilarge:
        length = 82
    elif dataset == datasetToLoad.wikismall:
        length = 82
    else:
        length = 67
    return min(length, restrict)
