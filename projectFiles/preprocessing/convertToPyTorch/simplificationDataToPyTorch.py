import torch
from torch.utils.data import DataLoader

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall
from projectFiles.seq2seq.constants import indices, SOS, EOS, device

def processPair(pair):
    originalIndices = pair['originalIndices'] + [EOS]
    simpleIndices = pair['simpleIndices'] + [EOS]

    originalTorch = torch.tensor(originalIndices, dtype=torch.long, device=device).view(-1, 1)
    simpleTorch = torch.tensor(simpleIndices, dtype=torch.long, device=device).view(-1, 1)

    return (originalTorch, simpleTorch)

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

    datasetProcessed = {"train": [processPair(val) for val in datasetLoaded.train],
                        "dev": [processPair(val) for val in datasetLoaded.dev],
                        "test": [processPair(val) for val in datasetLoaded.test]}

    return datasetProcessed

#simplificationDataToPyTorch(datasetToLoad.wikismall)
