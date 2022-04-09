import torch
from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.seq2seq.constants import EOS, device


class simplificationSet():
    def __init__(self, original, allSimple, dataset, language="en"):
        self.original = original
        self.allSimple = allSimple
        self.originalTokenized = original
        self.allSimpleTokenized = allSimple
        self._removeEscapedCharacters(dataset)
        self._makeReplacementsGEC()
        self._tokenise(dataset)
        self.dataset = dataset
        self.language = language
        self.originalIndices = None
        self.allSimpleIndices = None
        self.originalTorch = None
        self.allSimpleTorches = None
        self.predicted = ""

    def _tokenise(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.originalTokenized = self.originalTokenized.split(" ")
            self.allSimpleTokenized = [sentence.split(" ") for sentence in self.allSimpleTokenized]
        else:
            self.originalTokenized = word_tokenize(self.originalTokenized)
            self.allSimpleTokenized = [word_tokenize(sentence) for sentence in self.allSimpleTokenized]
        self.tokenized = True

    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        for (frm, to) in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"), ("-RCB-", "}"), ("-LAB-", "<"), ("-RAB-", ">")]:
            sentence = sentence.replace(frm, to)
        return sentence

    def _makeReplacementsGEC(self):
        for (frm, to) in [("''", '"'), ("--", "-"), ("`", "'")]:
            self.originalTokenized = self.originalTokenized.replace(frm, to)
            self.allSimpleTokenized = [sentence.replace(frm, to) for sentence in self.allSimpleTokenized]

    #Run before tokenisation
    def _removeEscapedCharacters(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.originalTokenized = self._removeEscapedBracketsFromSentenceWiki(self.originalTokenized)
            self.allSimpleTokenized = [self._removeEscapedBracketsFromSentenceWiki(sentence) for sentence in
                                       self.allSimpleTokenized]
        return

    def _safeSearch(self, indices, word, maxIndices=253401):
        try:
            return indices[word]
        except:
            # indices start at 0
            return maxIndices - 1

    def addIndices(self, indices, maxIndices=253401):
        self.originalIndices = [min(self._safeSearch(indices, word, maxIndices - 2), maxIndices - 3) + 2 for word in
                                self.originalTokenized]
        self.allSimpleIndices = [
            [min(self._safeSearch(indices, word, maxIndices - 2), maxIndices - 3) + 2 for word in sentence] for sentence
            in self.allSimpleTokenized]

    def getData(self):
        base = {"originalTokenized": self.originalTokenized, "allSimpleTokenized": self.allSimpleTokenized,
                "original": self.original,
                "allSimple": self.allSimple}
        if self.originalIndices:
            base["originalIndices"] = self.originalIndices
        if self.allSimpleIndices:
            base["allSimpleIndices"] = self.allSimpleIndices
        if self.predicted:
            base["predicted"] = self.predicted
        if self.originalTorch:
            base["originalTorch"] = self.originalTorch
        if self.allSimpleTorches:
            base["allSimpleTorches"] = self.allSimpleTorches

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

    # Returns a list, one for each
    def getMetric(self, lambdaFunc):
        return [lambdaFunc(self.originalTokenized, simplifiedSentence) for
                simplifiedSentence in self.allSimpleTokenized]
