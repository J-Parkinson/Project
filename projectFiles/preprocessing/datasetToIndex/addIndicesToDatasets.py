from collections import Counter

from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsala import loadNewsala
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall


def processDataset(dataset):
    dataset = dataset.dataset
    datasetOriginal = [sent.originalTokenized for sent in dataset]
    datasetSimple = [sent.allSimpleTokenized for sent in dataset]

    datasetOriginalFlatten = [word for sent in datasetOriginal for word in sent]
    datasetSimpleFlatten = [word for sents in datasetSimple for sent in sents for word in sent]

    datasetsBothFlatten = datasetOriginalFlatten + datasetSimpleFlatten
    counter = Counter(datasetsBothFlatten)
    return counter


asset = loadAsset()
print("asset")
newsala = loadNewsala()
print("newsala")
wikismall = loadWikiSmall()
print("wikismall")
wikilarge = loadWikiLarge()
print("wikilarge")

counter = Counter()

for dataset in [asset.train, asset.dev, asset.test,
                newsala.train, newsala.dev, newsala.test,
                wikismall.train, wikismall.dev, wikismall.test,
                wikilarge.train, wikilarge.dev, wikilarge.test]:
    counterToAdd = processDataset(dataset)
    counter += counterToAdd

words = [word[0] for word in counter.most_common()]

with open("indicesNew.txt", "w+", encoding="utf-8") as indexFile:
    dataToWrite = "\n".join(list(words))
    indexFile.write(dataToWrite)
