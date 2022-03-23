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

print("Max len sentence:",  max([max([len(sentence["original"]) for sentence in dataset]) for dataset in datasets]))
