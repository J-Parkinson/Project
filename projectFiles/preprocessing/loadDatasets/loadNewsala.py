from projectFiles.helpers.DatasetSplits import datasetSplits
from projectFiles.helpers.SimplificationData import *
from os import walk
import pickle

def loadNewsala(loadPickleFile=True, pickleFile=True, restrictLanguage="en", fullSimplifyOnly=False, startLoc=""):
    baseLoc = f"{startLoc}datasets/newsala"

    if loadPickleFile:
        return pickle.load(open(f"{baseLoc}/pickled.p", "rb"))
    dataset = datasetToLoad.newsala

    #Find names of every article in dataset
    allFilenames = tuple(walk(baseLoc))[0][2]

    #Larger numbers are more simplified
    uniqueFilenamesAndLangs = list(set([".".join(filename.split(".", 2)[:2]) for filename in allFilenames]))
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

        simplificationGroup = simplificationSet(allSimplifications[0], [allSimplifications[-1]] if fullSimplifyOnly else allSimplifications[1:], dataset, language=lang)

        datasetToAddTo = datasetSplits.train
        if len(allSimplifications) > 5:
            datasetToAddTo = datasetSplits.dev
        elif p % 25 == 0:
            datasetToAddTo = datasetSplits.test

        if datasetToAddTo == datasetSplits.train:
            everyPairTrain.append(simplificationGroup)
        elif datasetToAddTo == datasetSplits.dev:
            everyPairDev.append(simplificationGroup)
        else:
            everyPairTest.append(simplificationGroup)

    pairsTrain = simplificationDataset(everyPairTrain)
    pairsDev = simplificationDataset(everyPairDev)
    pairsTest = simplificationDataset(everyPairTest)

    dataset = simplificationDatasets(dataset, pairsTrain, pairsDev, pairsTest)

    if pickleFile:
        pickle.dump(dataset, open(f"{baseLoc}/pickled.p", "wb"))
    return dataset

#loadNewsala()