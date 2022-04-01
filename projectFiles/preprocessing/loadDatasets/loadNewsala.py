import pickle
from os import walk

from projectFiles.constants import baseLoc
from projectFiles.helpers.DatasetSplits import datasetSplits
from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData.SimplificationDataset import simplificationDataset
from projectFiles.helpers.SimplificationData.SimplificationDatasets import simplificationDatasets
from projectFiles.helpers.SimplificationData.SimplificationSet import simplificationSet


def loadNewsala(loadPickleFile=True, pickleFile=False, restrictLanguage="en", fullSimplifyOnly=False):
    print("Loading Newsala.")

    if loadPickleFile:
        print("Loading from file.")
        return pickle.load(open(f"{baseLoc}/datasets/newsala/pickled.p", "rb"))
    dataset = datasetToLoad.newsala

    # Find names of every article in dataset
    allFilenames = tuple(walk(f"{baseLoc}/datasets/newsala"))[0][2]

    #Larger numbers are more simplified
    uniqueFilenamesAndLangs = list(set([".".join(filename.split(".", 2)[:2]) for filename in allFilenames]))
    uniqueFilenames = [filename.split(".")[0] for filename in uniqueFilenamesAndLangs]
    uniqueLangs = [filename.split(".")[1] for filename in uniqueFilenamesAndLangs]

    uniqueFilenamesAndLangs = list(zip(uniqueFilenames, uniqueLangs))
    uniqueFilenamesAndLangs.sort(key=lambda setOf: setOf[0])

    everyPairTrain = []
    everyPairDev = []
    everyPairTest = []

    lenFile = len(uniqueFilenamesAndLangs)

    for p, (filename, lang) in enumerate(uniqueFilenamesAndLangs):
        if restrictLanguage and lang != restrictLanguage:
            continue

        allSimplificationsFiles = [open(f"{baseLoc}/datasets/newsala/{filename}.{lang}.{i}.txt", "r", encoding="utf-8")
                                   for i in range(5)]
        try:
            allSimplificationsFiles.append(open(f"{baseLoc}/datasets/newsala/{filename}.{lang}.5.txt"))
        except:
            pass

        allSimplifications = [file.read().split("\n\n") for file in allSimplificationsFiles]
        for file in allSimplificationsFiles:
            file.close()

        simplificationSets = list(zip(*allSimplifications))

        simplificationGroup = [simplificationSet(set[0], [set[-1]] if fullSimplifyOnly else set[1:], dataset, language=lang) for set in simplificationSets]

        datasetToAddTo = datasetSplits.train
        if len(allSimplifications) > 5:
            datasetToAddTo = datasetSplits.dev
        elif p % 50 == 0:
            print(f"{int(p / lenFile * 100)}% complete")
            datasetToAddTo = datasetSplits.test

        if datasetToAddTo == datasetSplits.train:
            everyPairTrain += simplificationGroup
        elif datasetToAddTo == datasetSplits.dev:
            everyPairDev += simplificationGroup
        else:
            everyPairTest += simplificationGroup

    print("File loaded.")

    pairsTrain = simplificationDataset(everyPairTrain)
    pairsDev = simplificationDataset(everyPairDev)
    pairsTest = simplificationDataset(everyPairTest)

    dataset = simplificationDatasets(dataset, pairsTrain, pairsDev, pairsTest)

    if pickleFile:
        print("Saving file.")
        pickle.dump(dataset, open(f"{baseLoc}/datasets/newsala/pickled.p", "wb"))
        print("File saved.")
    return dataset

#loadNewsala(loadPickleFile=False, pickleFile=True)