from projectFiles.helpers.DatasetToLoad import datasetToLoad
import pickle

from projectFiles.helpers.SimplificationData.SimplificationDataset import simplificationDataset
from projectFiles.helpers.SimplificationData.SimplificationDatasets import simplificationDatasets
from projectFiles.helpers.SimplificationData.SimplificationSet import simplificationSet


def loadAsset(loadPickleFile=True, pickleFile=False, startLoc="../../../"):
    print("Loading ASSET.")
    baseLoc = f"{startLoc}datasets/asset"

    if loadPickleFile:
        print("Loading from file.")
        return pickle.load(open(f"{baseLoc}/pickled.p", "rb"))

    dataset = datasetToLoad.asset
    with open(f'{baseLoc}/asset.valid.orig', 'r', encoding='utf-8') as validOrig:
        validOrig = validOrig.read().splitlines()
    validPairs = [validOrig]
    for i in range(10):
        with open(f'{baseLoc}/asset.valid.simp.{i}', 'r', encoding='utf-8') as validSimp:
            validSimp = validSimp.read().splitlines()
            validPairs.append(validSimp)
    validSetData = list(zip(*validPairs))
    validSets = [simplificationSet(data[0], data[1:], dataset, language="en") for data in validSetData]
    pairsTrain = validSets[:-30]
    pairsDev = validSets[-30:]

    with open(f'{baseLoc}/asset.test.orig', 'r', encoding='utf-8') as testOrig:
        testOrig = testOrig.read().splitlines()
    testPairs = [testOrig]
    for i in range(10):
        with open(f'{baseLoc}/asset.test.simp.{i}', 'r', encoding='utf-8') as testSimp:
            testSimp = testSimp.read().splitlines()
            testPairs.append(testSimp)
    testSetData = list(zip(*testPairs))
    pairsTest = [simplificationSet(data[0], data[1:], dataset, language="en") for data in testSetData]
    pairsTrain = simplificationDataset(pairsTrain)
    pairsDev = simplificationDataset(pairsDev)
    pairsTest = simplificationDataset(pairsTest)

    dataset = simplificationDatasets(dataset, pairsTrain, pairsDev, pairsTest)

    print("Dataset loaded.")

    if pickleFile:
        print("Saving dataset.")
        pickle.dump(dataset, open(f"{baseLoc}/pickled.p", "wb"))
        print("Dataset saved.")
    return dataset

#loadAsset(False, True)