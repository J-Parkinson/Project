from enum import Enum

class datasetToLoad(Enum):
    asset = 0
    wikismall = 1
    wikilarge = 2
    newsala = 3

def dsName(dataset):
    return ["asset", "wikiSmall", "wikiLarge", "newsala"][dataset.value]

def name2DTL(name):
    return [datasetToLoad.asset, datasetToLoad.wikismall, datasetToLoad.wikilarge, datasetToLoad.newsala][
        ["asset", "wikiSmall", "wikiLarge", "newsala"].index(name)]
