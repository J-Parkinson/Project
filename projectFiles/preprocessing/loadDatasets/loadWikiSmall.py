import pickle

from projectFiles.constants import baseLoc
from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData import *


def loadWikiSmall(loadPickleFile=True, pickleFile=False, isAnonymised=False):
    if loadPickleFile:
        return pickle.load(open(f"{baseLoc}/datasets/wikismall/pickled{'ori' if not isAnonymised else ''}.p", "rb"))

    startLoc = baseLoc + f"/datasets/wikismall/PWKP_108016.tag.80.aner{'.ori' if not isAnonymised else ''}"
    dataset = datasetToLoad.wikismall
    with open(f'{startLoc}.train.src', 'r', encoding='utf-8') as trainOrig:
        trainOrig = trainOrig.read().splitlines()
    trainPairs = []
    with open(f'{startLoc}.train.dst', 'r', encoding='utf-8') as trainSimp:
        trainSimp = trainSimp.read().splitlines()
        setOfTrainPairs = [simplificationSet(trainOrigElem, [trainSimpElem], dataset) for trainOrigElem, trainSimpElem
                           in zip(trainOrig, trainSimp)]
        trainPairs += setOfTrainPairs

    with open(f'{startLoc}.valid.src', 'r', encoding='utf-8') as validOrig:
        validOrig = validOrig.read().splitlines()
    validPairs = []
    with open(f'{startLoc}.valid.dst', 'r', encoding='utf-8') as validSimp:
        validSimp = validSimp.read().splitlines()
        setOfValidPairs = [simplificationSet(validOrigElem, [validSimpElem], dataset) for validOrigElem, validSimpElem
                           in zip(validOrig, validSimp)]
        validPairs += setOfValidPairs

    with open(f'{startLoc}.test.src', 'r', encoding='utf-8') as testOrig:
        testOrig = testOrig.read().splitlines()
    testPairs = []
    with open(f'{startLoc}.test.dst', 'r', encoding='utf-8') as testSimp:
        testSimp = testSimp.read().splitlines()
        setOfTestPairs = [simplificationSet(testOrigElem, [testSimpElem], dataset) for testOrigElem, testSimpElem in
                          zip(testOrig, testSimp)]
        testPairs += setOfTestPairs

    trainPairs = simplificationDataset(trainPairs)
    validPairs = simplificationDataset(validPairs)
    testPairs = simplificationDataset(testPairs)

    dataset = simplificationDatasets(dataset, trainPairs, validPairs, testPairs)

    if pickleFile:
        pickle.dump(dataset, open(f"{baseLoc}/datasets/wikismall/pickled{'ori' if not isAnonymised else ''}.p", "wb"))

    return dataset

#loadWikiSmall(loadPickleFile=False, pickleFile=True, isAnonymised=False, startLoc="../../../")
#loadWikiSmall(loadPickleFile=False, pickleFile=True, isAnonymised=True, startLoc="../../../")