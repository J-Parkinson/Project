from collections import Counter

from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall

asset = loadAsset()
print("asset")
newsala = loadNewsala()
print("newsala")
wikismall = loadWikiSmall()
print("wikismall")
wikilarge = loadWikiLarge()
print("wikilarge")

datasets = [asset.train, asset.dev, asset.test,
            newsala.train, newsala.dev, newsala.test,
            wikismall.train, wikismall.dev, wikismall.test,
            wikilarge.train, wikilarge.dev, wikilarge.test]

datasets = list(map(lambda x: x.dataset, datasets))
datasetsOriginal = list(map(lambda x: list(map(lambda y: y.originalTokenized, x)), datasets))
datasetsSimple = list(map(lambda x: list(map(lambda y: y.allSimpleTokenized, x)), datasets))

datasetsOriginal = [item.lower() for sublist in datasetsOriginal for subsublist in sublist for item in subsublist]
datasetsSimple = [item.lower() for sublist in datasetsSimple for subsublist in sublist for item in subsublist]

datasetsOriginal = Counter(datasetsOriginal)
datasetsSimple = Counter(datasetsSimple)
datasets = datasetsOriginal + datasetsSimple
datasets = [x[0] for x in datasets.most_common()]

with open("indices.txt", "w+", encoding="utf-8") as indexFile:
    dataToWrite = "\n".join(list(datasets))
    indexFile.write(dataToWrite)
