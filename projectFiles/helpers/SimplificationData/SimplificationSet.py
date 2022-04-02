import torch
from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.seq2seq.constants import EOS, device


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
        self.originalIndices = []
        self.allSimpleIndices = []
        self.originalTorch = None
        self.allSimpleTorches = None
        self.predicted = ""

    def _tokenise(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self.original.split(" ")
            self.allSimple = [sentence.split(" ") for sentence in self.allSimple]
        else:
            self.original = word_tokenize(self.original)
            self.allSimple = [word_tokenize(sentence) for sentence in self.allSimple]
        self.tokenized = True

    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        for (frm, to) in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"), ("-RCB-", "}"), ("-LAB-", "<"), ("-RAB-", ">")]:
            sentence = sentence.replace(frm, to)
        return sentence

    def _makeReplacementsGEC(self):
        for (frm, to) in [("''", '"'), ("--", "-"), ("`", "'")]:
            self.original = self.original.replace(frm, to)
            self.allSimple = [sentence.replace(frm, to) for sentence in self.allSimple]

    #Run before tokenisation
    def _removeEscapedCharacters(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self._removeEscapedBracketsFromSentenceWiki(self.original)
            self.allSimple = [self._removeEscapedBracketsFromSentenceWiki(sentence) for sentence in self.allSimple]
        return

    def _safeSearch(self, indices, word, maxIndices=222823):
        try:
            return indices[word.lower()]
        except:
            # indices start at 0
            return maxIndices - 1

    def addIndices(self, indices, maxIndices=222823):
        self.originalIndices = [min(self._safeSearch(indices, word, maxIndices - 2), maxIndices - 3) + 2 for word in
                                self.original]
        self.allSimpleIndices = [
            [min(self._safeSearch(indices, word, maxIndices - 2), maxIndices - 3) + 2 for word in sentence] for sentence
            in self.allSimple]

    def getIndices(self):
        if self.originalIndices and self.allSimpleIndices:
            return {"original": self.originalIndices, "simple": self.allSimpleIndices}
        else:
            return None

    def getData(self):
        return {"original": self.original, "allSimple": self.allSimple,
                "originalIndices": self.originalIndices, "allSimpleIndices": self.allSimpleIndices}

    def torchSet(self):
        originalIndices = self.originalIndices + [EOS]
        simpleIndices = [pairVal + [EOS] for pairVal in self.allSimpleIndices]

        originalTorch = torch.tensor(originalIndices, dtype=torch.long, device=device).view(-1, 1)
        simpleTorches = [torch.tensor(simpleIndex, dtype=torch.long, device=device).view(-1, 1) for simpleIndex in
                         simpleIndices]

        self.originalTorch = originalTorch
        self.allSimpleTorches = simpleTorches

        return (originalTorch, simpleTorches)

    def addPredicted(self, prediction):
        self.predicted = prediction

    def getPredicted(self):
        return self.predicted

    # Returns a list, one for each
    def getMetric(self, lambdaFunc):
        return [lambdaFunc(self.original, simplifiedSentence, self.originalIndices, simplifiedIndices) for
                simplifiedSentence, simplifiedIndices in zip(self.allSimple, self.allSimpleIndices)]
