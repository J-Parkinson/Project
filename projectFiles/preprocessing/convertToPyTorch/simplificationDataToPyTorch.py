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
    print(originalTorch)
    print(pair['original'])

    return (originalTorch, simpleTorch)

def simplificationDataToPyTorch(dataset):
    if dataset == datasetToLoad.asset:
        datasetLoaded = loadAsset(startLoc="../../../")
    elif dataset == datasetToLoad.newsala:
        datasetLoaded = loadNewsala(startLoc="../../../")
    elif dataset == datasetToLoad.wikilarge:
        datasetLoaded = loadWikiLarge(startLoc="../../../")
    else:
        datasetLoaded = loadWikiSmall(startLoc="../../../")

    datasetLoaded.addIndices(indices)

    datasetProcessed = {"train": [processPair(val) for val in datasetLoaded.train],
                        "dev": [processPair(val) for val in datasetLoaded.dev],
                        "test": [processPair(val) for val in datasetLoaded.test]}

    print(datasetProcessed)


simplificationDataToPyTorch(datasetToLoad.wikismall)
