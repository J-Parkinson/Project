import os

from helpers.DatasetSplits import datasetSplits
from helpers.SimplificationData import *
from helpers.DatasetToLoad import *
from os import walk

def loadNewsala(restrictLanguage=None, restrictMinDiffBetween=None, fullSimplifyOnly=False):
    baseLoc = "../../datasets/newsala"
    dataset = datasetToLoad.newsala

    #Find names of every article in dataset
    allFilenames = tuple(walk(baseLoc))[0][2]

    #Larger numbers are more simplified
    uniqueFilenamesAndLangs = list(set([".".join(filename.split(".", 2)[:2]) for filename in allFilenames]))
    print(uniqueFilenamesAndLangs)
    uniqueFilenames = [filename.split(".")[0] for filename in uniqueFilenamesAndLangs]
    uniqueLangs = [filename.split(".")[1] for filename in uniqueFilenamesAndLangs]

    uniqueFilenamesAndLangs = list(zip(uniqueFilenames, uniqueLangs))
    uniqueFilenamesAndLangs.sort(key=lambda setOf: setOf[0])

    everyPairTrain = []
    everyPairDev = []
    everyPairTest = []

    for p, (filename, lang) in enumerate(uniqueFilenamesAndLangs):
        if restrictLanguage and lang != restrictLanguage:
            continue

        allSimplificationsFiles = [open(f"{baseLoc}/{filename}.{lang}.{i}.txt", "r", encoding="utf-8") for i in range(5)]
        try:
            allSimplificationsFiles.append(open(f"{baseLoc}/{filename}.{lang}.5.txt"))
        except:
            pass

        allSimplifications = [file.read().split("\n\n") for file in allSimplificationsFiles]
        for file in allSimplificationsFiles:
            file.close()

        allPairs = []
        for i in range(len(allSimplifications) - 1):
            for j in range(i+1, len(allSimplifications)):
                if restrictMinDiffBetween and (j-i) <= restrictMinDiffBetween:
                    continue
                if fullSimplifyOnly and j != len(allSimplifications) - 1 and i != 0:
                    continue
                original = allSimplifications[i]
                simplified = allSimplifications[j]
                #Here we make all pairs for a given simplification level pair
                pairs = [simplificationPair(orig, simp, dataset, simplicityFactor=(i,j), language=lang) for orig, simp in zip(original, simplified)]
                allPairs += pairs

        datasetToAddTo = datasetSplits.train
        if len(allSimplifications) > 5:
            datasetToAddTo = datasetSplits.dev
        elif p % 25 == 0:
            datasetToAddTo = datasetSplits.test

        if datasetToAddTo == datasetSplits.train:
            everyPairTrain += allPairs
        elif datasetToAddTo == datasetSplits.dev:
            everyPairDev += allPairs
        else:
            everyPairTest += allPairs

    dataset = simplificationDataset(dataset, everyPairTrain, everyPairDev, everyPairTest)
    return dataset

loadNewsala()