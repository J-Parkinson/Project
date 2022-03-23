from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall
from collections import Counter

asset = loadAsset(startLoc="../../../")
print("asset")
newsala = loadNewsala(startLoc="../../../")
print("newsala")
wikismall = loadWikiSmall(startLoc="../../../")
print("wikismall")
wikilarge = loadWikiLarge(startLoc="../../../")
print("wikilarge")

datasets = [asset.train, asset.dev, asset.test,
            newsala.train, newsala.dev, newsala.test,
            wikismall.train, wikismall.dev, wikismall.test,
            wikilarge.train, wikilarge.dev, wikilarge.test]

datasets = list(map(lambda x: x.dataset, datasets))
datasetsOriginal = list(map(lambda x: list(map(lambda y: y.original, x)), datasets))
datasetsSimple = list(map(lambda x: list(map(lambda y: y.simple, x)), datasets))

datasetsOriginal = [item.lower() for sublist in datasetsOriginal for subsublist in sublist for item in subsublist]
datasetsSimple = [item.lower() for sublist in datasetsSimple for subsublist in sublist for item in subsublist]

datasetsOriginal = Counter(datasetsOriginal)
datasetsSimple = Counter(datasetsSimple)
datasets = datasetsOriginal + datasetsSimple
datasets = [x[0] for x in datasets.most_common()]

with open("indices.txt", "w+", encoding="utf-8") as indexFile:
    dataToWrite = "\n".join(list(datasets))
    indexFile.write(dataToWrite)
