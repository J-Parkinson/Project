import pickle

from projectFiles.constants import baseLoc
from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData.SimplificationDataset import simplificationDataset
from projectFiles.helpers.SimplificationData.SimplificationDatasets import simplificationDatasets
from projectFiles.helpers.SimplificationData.SimplificationSet import simplificationSet


def loadAsset(loadPickleFile=True, pickleFile=False):
    print("Loading ASSET.")

    if loadPickleFile:
        print("Loading from file.")
        return pickle.load(open(f"{baseLoc}/datasets/asset/pickled.p", "rb"))

    dataset = datasetToLoad.asset
    with open(f'{baseLoc}/datasets/asset/asset.valid.orig', 'r', encoding='utf-8') as validOrig:
        validOrig = validOrig.read().splitlines()
    validPairs = [validOrig]
    for i in range(10):
        with open(f'{baseLoc}/datasets/asset/asset.valid.simp.{i}', 'r', encoding='utf-8') as validSimp:
            validSimp = validSimp.read().splitlines()
            validPairs.append(validSimp)
    validSetData = list(zip(*validPairs))
    validSets = [simplificationSet(data[0], data[1:], dataset, language="en") for data in validSetData]
    pairsTrain = validSets[:-100]
    pairsDev = validSets[-100:]

    with open(f'{baseLoc}/datasets/asset/asset.test.orig', 'r', encoding='utf-8') as testOrig:
        testOrig = testOrig.read().splitlines()
    testPairs = [testOrig]
    for i in range(10):
        with open(f'{baseLoc}/datasets/asset/asset.test.simp.{i}', 'r', encoding='utf-8') as testSimp:
            testSimp = testSimp.read().splitlines()
            testPairs.append(testSimp)
    testSetData = list(zip(*testPairs))
    pairsTest = [simplificationSet(data[0], data[1:], dataset, language="en") for data in testSetData]
    pairsTrain = simplificationDataset(pairsTrain, initialiseCL=not pickleFile)
    pairsDev = simplificationDataset(pairsDev, initialiseCL=not pickleFile)
    pairsTest = simplificationDataset(pairsTest, initialiseCL=not pickleFile)

    dataset = simplificationDatasets(dataset, pairsTrain, pairsDev, pairsTest)

    print("Dataset loaded.")

    if pickleFile:
        print("Saving dataset.")
        pickle.dump(dataset, open(f"{baseLoc}/datasets/asset/pickled.p", "wb"))
        print("Dataset saved.")
    return dataset

#loadAsset(False, True)