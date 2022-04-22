from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch

# Calculates training, development and test dataset sizes (in terms of number of simplified sentences) for each dataset

sizes = {datasetToLoad.asset: (19000, 1000, 3590), datasetToLoad.newsela: (133720, 3260, 2616),
         datasetToLoad.wikismall: (88837, 205, 100), datasetToLoad.wikilarge: (296402, 992, 359)}


def calculateTrainDevTestSizesForDataset(data, datasetName):
    # print(data.train[0]['simplified'])
    print(f"Training set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.train)}\n"
          f"\tNumber of pairs of sentences: {sum([len(val.allSimpleTokenized) for val in data.train])}")
    print(f"Development set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.dev)}\n"
          f"\tNumber of pairs of sentences: {sum([len(val.allSimpleTokenized) for val in data.dev])}")
    print(f"Test set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.test)}\n"
          f"\tNumber of pairs of sentences: {sum([len(val.allSimpleTokenized) for val in data.test])}")
    return


def calculateTrainDevTestSizes():
    asset = simplificationDataToPyTorch(datasetToLoad.asset)
    calculateTrainDevTestSizesForDataset(asset, "asset")

    newsela = simplificationDataToPyTorch(datasetToLoad.newsela)
    calculateTrainDevTestSizesForDataset(newsela, "newsela")

    wikismall = simplificationDataToPyTorch(datasetToLoad.wikismall)
    calculateTrainDevTestSizesForDataset(wikismall, "wikiSmall")

    wikilarge = simplificationDataToPyTorch(datasetToLoad.wikilarge)
    calculateTrainDevTestSizesForDataset(wikilarge, "wikiLarge")


calculateTrainDevTestSizes()
