from helpers.Anonymisation import anonymisation
from helpers.SimplificationData import *
from helpers.DatasetToLoad import *

def loadWikiLarge(isAnonymised):
    baseLoc = f"../../datasets/wikilarge/wiki.full.aner{'.ori' if isAnonymised.value else ''}"
    dataset = datasetToLoad.wikilarge
    with open(f'{baseLoc}.train.src', 'r', encoding='utf-8') as trainOrig:
        trainOrig = trainOrig.read().splitlines()
    trainPairs = []
    with open(f'{baseLoc}.train.dst', 'r', encoding='utf-8') as trainSimp:
        trainSimp = trainSimp.read().splitlines()
        setOfTrainPairs = [simplificationPair(trainOrigElem, trainSimpElem, dataset) for trainOrigElem, trainSimpElem in zip(trainOrig, trainSimp)]
        trainPairs += setOfTrainPairs

    with open(f'{baseLoc}.valid.src', 'r', encoding='utf-8') as validOrig:
        validOrig = validOrig.read().splitlines()
    validPairs = []
    with open(f'{baseLoc}.valid.dst', 'r', encoding='utf-8') as validSimp:
        validSimp = validSimp.read().splitlines()
        setOfValidPairs = [simplificationPair(validOrigElem, validSimpElem, dataset) for validOrigElem, validSimpElem in zip(validOrig, validSimp)]
        validPairs += setOfValidPairs

    with open(f'{baseLoc}.test.src', 'r', encoding='utf-8') as testOrig:
        testOrig = testOrig.read().splitlines()
    testPairs = []
    with open(f'{baseLoc}.test.dst', 'r', encoding='utf-8') as testSimp:
        testSimp = testSimp.read().splitlines()
        setOfTestPairs = [simplificationPair(testOrigElem, testSimpElem, dataset) for testOrigElem, testSimpElem in zip(testOrig, testSimp)]
        testPairs += setOfTestPairs

    dataset = simplificationDataset(dataset, trainPairs, validPairs, testPairs)
    return dataset

x = loadWikiLarge(anonymisation.original).train[0]
print(x.original)
print(x.simple)