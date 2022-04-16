from collections import Counter

from matplotlib import pyplot as plt

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch

asset = simplificationDataToPyTorch(datasetToLoad.asset, embeddingType.bert)
print("asset")
newsala = simplificationDataToPyTorch(datasetToLoad.newsala, embeddingType.bert)
print("newsala")
wikismall = simplificationDataToPyTorch(datasetToLoad.wikismall, embeddingType.bert)
print("wikismall")
wikilarge = simplificationDataToPyTorch(datasetToLoad.wikilarge, embeddingType.bert)
print("wikilarge")

datasets = [asset.train, asset.dev, asset.test,
            newsala.train, newsala.dev, newsala.test,
            wikismall.train, wikismall.dev, wikismall.test,
            wikilarge.train, wikilarge.dev, wikilarge.test]

maxSentence = 0
allLensOrig = Counter()
allLensSimp = Counter()
for dataset in datasets:
    viewsSet = [setV.getAllViews() for setV in dataset]
    views = [view for views in viewsSet for view in views]
    allOrig = [len(view.originalIndices) for view in views]
    allLensOrig.update(allOrig)
    maxOrig = max(allOrig)
    allSimp = [len(view.simpleIndices) for view in views]
    allLensSimp.update(allSimp)
    maxSimp = max(allSimp)
    maxSentence = max([maxOrig, maxSimp, maxSentence])

print(f"Max sentence: {maxSentence}")
plt.title("Lengths of sentences across all datasets including both original and simplified")
allLens = allLensOrig + allLensSimp
plt.plot(allLens.items())
plt.show()
