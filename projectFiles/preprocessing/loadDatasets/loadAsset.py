from projectFiles.helpers.SimplificationData import *


def loadAsset():
    baseLoc = "../../datasets/asset"
    dataset = datasetToLoad.asset
    with open(f'{baseLoc}/asset.valid.orig', 'r', encoding='utf-8') as validOrig:
        validOrig = validOrig.read().splitlines()
    validPairs = []
    for i in range(10):
        with open(f'{baseLoc}/asset.valid.simp.{i}', 'r', encoding='utf-8') as validSimp:
            validSimp = validSimp.read().splitlines()
            setOfValidPairs = [simplificationPair(validOrigElem, validSimpElem, dataset) for validOrigElem, validSimpElem in zip(validOrig, validSimp)]
            validPairs += setOfValidPairs
    pairsTrain = validPairs[:-30]
    pairsDev = validPairs[-30:]

    with open(f'{baseLoc}/asset.test.orig', 'r', encoding='utf-8') as testOrig:
        testOrig = testOrig.read().splitlines()
    pairsTest = []
    for i in range(10):
        with open(f'{baseLoc}/asset.test.simp.{i}', 'r', encoding='utf-8') as testSimp:
            testSimp = testSimp.read().splitlines()
            setOfTestPairs = [simplificationPair(testOrigElem, testSimpElem, dataset) for testOrigElem, testSimpElem in zip(testOrig, testSimp)]
            pairsTest += setOfTestPairs

    pairsTrain = simplificationDataset(pairsTrain)
    pairsDev = simplificationDataset(pairsDev)
    pairsTest = simplificationDataset(pairsTest)

    dataset = simplificationDatasets(dataset, pairsTrain, pairsDev, pairsTest)
    return dataset

x = loadAsset().train[0]
print(x.original)
print(x.simple)