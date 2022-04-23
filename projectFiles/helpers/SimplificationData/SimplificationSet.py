import torch

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData.SimplificationSetView import simplificationSetView
from projectFiles.seq2seq.constants import device


class simplificationSet():
    def __init__(self, original, allSimple, dataset, language="en"):
        self.original = original
        self.allSimple = allSimple
        self.originalTokenized = original
        self.allSimpleTokenized = allSimple
        self._removeEscapedCharacters(dataset)
        self._makeReplacementsGEC()
        self.dataset = dataset
        self.language = language
        self.predicted = ""
        self.originalIndices = None
        self.allSimpleIndices = None
        self.originalTorch = None
        self.allSimpleTorches = None

    # WikiSmall/Large escapes brackets and arrows, which we add back in
    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        for (frm, to) in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"), ("-RCB-", "}"), ("-LAB-", "<"), ("-RAB-", ">")]:
            sentence = sentence.replace(frm, to)
        return sentence

    #Gector (what another text simplification model is based on) suggests replacing these strings in the text
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

    def addPredicted(self, prediction):
        self.predicted = prediction

    # Returns a list, one for each simplified sentence
    # Returning indices rather than sorting the list directly enables us to easily reverse the curriculum learning
    # process (else we would need to do original sentence matching in order to convert simplificationViews back into
    # simplificationSets
    def getMetric(self, lambdaFunc):
        return [lambdaFunc(self.originalTokenized, simplifiedSentence) for
                simplifiedSentence in self.allSimpleTokenized]

    # Creates torch tensors from each sentence indices - for use by Cuda (`device'-depending)
    def torchSet(self):
        self.originalTorch = torch.tensor(self.originalIndices, dtype=torch.long, device=device).view(-1, 1)
        self.allSimpleTorches = [torch.tensor(simpleIndex, dtype=torch.long, device=device).view(-1, 1) for simpleIndex
                                 in self.allSimpleIndices]

        return (self.originalTorch, self.allSimpleTorches)

    #Returns view from sentence in simplification set
    def getView(self, yIndex):
        return simplificationSetView(self.original,
                                     self.allSimple[yIndex],
                                     self.originalTokenized,
                                     self.allSimpleTokenized[yIndex],
                                     self.dataset,
                                     self.language,
                                     self.predicted,
                                     self.originalIndices,
                                     None if not self.allSimpleIndices else self.allSimpleIndices[yIndex],
                                     self.originalTorch,
                                     None if not self.allSimpleIndices else self.allSimpleTorches[yIndex])

    def getAllViews(self):
        return [self.getView(yIndex) for yIndex in range(len(self.allSimple))]
