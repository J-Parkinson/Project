from collections import Counter

from matplotlib import pyplot as plt

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch

newsela = simplificationDataToPyTorch(datasetToLoad.newsela, embeddingType.indices)
print("newsela")

datasets = [newsela.train, newsela.dev, newsela.test]


def safeCheck(x, y):
    return x[y] if x[y] else 0


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
print(sum(allLensSimp.values()))
allLensSorted = list(allLens.items())
allLensSorted.sort(key=lambda x: x[0])
allLensX, allLensY = [list(i) for i in zip(*allLensSorted)]
plt.plot(allLensX, allLensY)
plt.show()
