import torch
from torch.utils.data import DataLoader

from projectFiles.helpers.DatasetSplits import datasetSplits
from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall
from projectFiles.seq2seq.constants import indices, SOS, EOS, device

def simplificationDataToPyTorch(dataset, maxIndices=75000, startLoc="../../../"):
    if dataset == datasetToLoad.asset:
        print("Loading ASSET")
        datasetLoaded = loadAsset(startLoc=startLoc)
    elif dataset == datasetToLoad.newsala:
        print("Loading Newsala")
        datasetLoaded = loadNewsala(startLoc=startLoc)
    elif dataset == datasetToLoad.wikilarge:
        print("Loading WikiLarge")
        datasetLoaded = loadWikiLarge(startLoc=startLoc)
    else:
        print("Loading WikiSmall")
        datasetLoaded = loadWikiSmall(startLoc=startLoc)

    datasetLoaded.addIndices(indices, maxIndices=maxIndices)

    torchObjects = datasetLoaded.torchProcess()

    return torchObjects, datasetLoaded

#simplificationDataToPyTorch(datasetToLoad.asset)
