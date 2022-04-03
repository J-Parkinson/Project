from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall
from projectFiles.seq2seq.constants import indices


def simplificationDataToPyTorch(dataset, maxIndices=222823):
    if dataset == datasetToLoad.asset:
        print("Loading ASSET")
        datasetLoaded = loadAsset()
    elif dataset == datasetToLoad.newsala:
        print("Loading Newsala")
        datasetLoaded = loadNewsala()
    elif dataset == datasetToLoad.wikilarge:
        print("Loading WikiLarge")
        datasetLoaded = loadWikiLarge()
    else:
        print("Loading WikiSmall")
        datasetLoaded = loadWikiSmall()

    datasetLoaded.addIndices(indices, maxIndices=maxIndices)

    datasetLoaded.torchProcess()

    return datasetLoaded

#simplificationDataToPyTorch(datasetToLoad.asset)
