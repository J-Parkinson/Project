from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch


def calculateTrainDevTestSizesForDataset(data, datasetName):
    # print(data.train[0]['simplified'])
    print(f"Training set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.train)}\n"
          f"\tNumber of pairs of sentences: {sum([len([val['simplified'] for val in data.train])])}")
    print(f"Development set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.dev)}\n"
          f"\tNumber of pairs of sentences: {sum([len([val['simplified'] for val in data.dev])])}")
    print(f"Test set size of {datasetName}:\n"
          f"\tNumber of sets of sentences: {len(data.test)}\n"
          f"\tNumber of pairs of sentences: {sum([len([val['simplified'] for val in data.test])])}")
    return


def calculateTrainDevTestSizes():
    _, asset = simplificationDataToPyTorch(datasetToLoad.asset)
    calculateTrainDevTestSizesForDataset(asset, "asset")

    _, newsala = simplificationDataToPyTorch(datasetToLoad.newsala)
    calculateTrainDevTestSizesForDataset(newsala, "newsala")

    _, wikismall = simplificationDataToPyTorch(datasetToLoad.wikismall)
    calculateTrainDevTestSizesForDataset(wikismall, "wikiSmall")

    _, wikilarge = simplificationDataToPyTorch(datasetToLoad.wikilarge)
    calculateTrainDevTestSizesForDataset(wikilarge, "wikiLarge")


calculateTrainDevTestSizes()
