import torch
from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.seq2seq.constants import EOS, device


class simplificationSetNLTK(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self._tokenise(dataset)
        self.originalIndices = None
        self.allSimpleIndices = None
        self.originalTorch = None
        self.allSimpleTorches = None

    def _tokenise(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.originalTokenized = self.originalTokenized.split(" ")
            self.allSimpleTokenized = [sentence.split(" ") for sentence in self.allSimpleTokenized]
        else:
            self.originalTokenized = word_tokenize(self.originalTokenized)
            self.allSimpleTokenized = [word_tokenize(sentence) for sentence in self.allSimpleTokenized]

    def addIndices(self, indices):
        self.originalIndices = [indices[word] for word in self.originalTokenized]
        self.allSimpleIndices = [[indices[word] for word in sentence] for sentence in self.allSimpleTokenized]

    def torchSet(self):
        originalIndices = self.originalIndices + [EOS]
        simpleIndices = [pairVal + [EOS] for pairVal in self.allSimpleIndices]

        self.originalTorch = torch.tensor(originalIndices, dtype=torch.long, device=device).view(-1, 1)
        self.allSimpleTorches = [torch.tensor(simpleIndex, dtype=torch.long, device=device).view(-1, 1) for simpleIndex
                                 in
                                 simpleIndices]

        return (self.originalTorch, self.allSimpleTorches)
