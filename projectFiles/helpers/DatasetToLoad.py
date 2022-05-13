from enum import Enum

class datasetToLoad(Enum):
    asset = 0
    wikismall = 1
    wikilarge = 2
    newsela = 3

def dsName(dataset):
    return ["asset", "wikiSmall", "wikiLarge", "newsela"][dataset.value]


def name2DTL(name):
    return [datasetToLoad.asset, datasetToLoad.wikismall, datasetToLoad.wikilarge, datasetToLoad.newsela][
        ["asset", "wikiSmall", "wikiLarge", "newsela"].index(name)]
