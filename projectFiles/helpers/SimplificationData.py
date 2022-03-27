import torch

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

class simplificationSet():
    def __init__(self, original, allSimple, dataset, language="en"):
        self.original = original
        self.allSimple = allSimple
        self._removeEscapedCharacters(dataset)
        self._makeReplacementsGEC()
        self._tokenise(dataset)
        self.dataset = dataset
        self.calculatedSimplicityFactor = 0
        self.language = language
        self.originalIndices = None
        self.simpleIndices = None

    def _tokenise(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self.original.split(" ")
            self.allSimple = self.allSimple.split(" ")
        else:
            self.original = word_tokenize(self.original)
            self.allSimple = word_tokenize(self.allSimple)
        self.tokenized = True

    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        for (frm, to) in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"), ("-RCB-", "}"), ("-LAB-", "<"), ("-RAB-", ">")]:
            sentence = sentence.replace(frm, to)
        return sentence

    def _makeReplacementsGEC(self):
        for (frm, to) in [("''", '"'), ("--", "-"), ("`", "'")]:
            self.original = self.original.replace(frm, to)
            self.allSimple = self.allSimple.replace(frm, to)

    #Run before tokenisation
    def _removeEscapedCharacters(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self._removeEscapedBracketsFromSentenceWiki(self.original)
            self.allSimple = self._removeEscapedBracketsFromSentenceWiki(self.allSimple)
        return

    def _safeSearch(self, indices, word, maxIndices=75000):
        try:
            return indices[word.lower()]
        except:
            #indices start at 0
            return maxIndices - 1

    def addIndices(self, indices, maxIndices=75000):
        self.originalIndices = [min(self._safeSearch(indices, word, maxIndices-2), maxIndices-3)+2 for word in self.original]
        self.simpleIndices = [min(self._safeSearch(indices, word, maxIndices-2), maxIndices-3) + 2 for word in self.allSimple]

    def getIndices(self):
        if self.originalIndices and self.simpleIndices:
            return {"original": self.originalIndices, "simple": self.simpleIndices}

    def getData(self):
        return {"original": self.original, "simplified": self.allSimple,
                "originalIndices": self.originalIndices, "simpleIndices": self.simpleIndices}

class simplificationDataset(Dataset):
    def _sortBySimplificationAmount(self):
        return

    def __init__(self, simplificationPairSet, sortBySimplificationAmount=False):
        self.dataset = simplificationPairSet
        if sortBySimplificationAmount:
            self._sortBySimplificationAmount()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].getData()

class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def addIndices(self, indices, maxIndices=75000):
        for pair in self.train.dataset:
            pair.addIndices(indices, maxIndices)
        for pair in self.dev.dataset:
            pair.addIndices(indices, maxIndices)
        for pair in self.test.dataset:
            pair.addIndices(indices, maxIndices)