from projectFiles.helpers.DatasetToLoad import datasetToLoad


def getMaxLens(dataset, restrict=None):
    if dataset == datasetToLoad.wikilarge:
        length = 82
    elif dataset == datasetToLoad.wikismall:
        length = 82
    else:
        length = 67
    return length if not restrict else min(length, restrict)
