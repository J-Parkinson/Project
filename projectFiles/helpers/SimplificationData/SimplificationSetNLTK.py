import torch
from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import getIndex
from projectFiles.seq2seq.constants import device


class simplificationSetNLTK(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self.tokenise()
        self.addIndices()

    def tokenise(self):
        if self.dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.originalTokenized = self.originalTokenized.split(" ")
            self.allSimpleTokenized = [sentence.split(" ") for sentence in self.allSimpleTokenized]
        else:
            self.originalTokenized = word_tokenize(self.originalTokenized)
            self.allSimpleTokenized = [word_tokenize(sentence) for sentence in self.allSimpleTokenized]
        # ADD SOS/EOS
        self.originalTokenized = ["<sos>"] + self.originalTokenized + ["<eos>"]
        self.allSimpleTokenized = [["<sos>"] + sentence + ["<eos>"] for sentence in self.allSimpleTokenized]
        self.originalTokenizedPadded = self._addPadding(self.originalTokenized)
        self.allSimpleTokenizedPadded = [self._addPadding(sentence) for sentence in self.allSimpleTokenized]

    def _addPadding(self, sentence):
        return sentence + ["<pad>" for _ in range(self.maxSentenceLen - len(sentence))]

    def addIndices(self):
        # padding added
        self.originalIndices = [getIndex(word) for word in self._addPadding(self.originalTokenized)]
        self.allSimpleIndices = [[getIndex(word) for word in self._addPadding(sentence)] for sentence in
                                 self.allSimpleTokenized]

    # Creates torch tensors from each sentence indices - for use by Cuda (`device'-depending)
    def torchSet(self):
        self.originalTorch = torch.tensor(self.originalIndices, dtype=torch.int64, device=device).view(-1, 1)
        self.allSimpleTorches = [torch.tensor(simpleIndex, dtype=torch.int64, device=device).view(-1, 1) for simpleIndex
                                 in self.allSimpleIndices]

        return (self.originalTorch, self.allSimpleTorches)
