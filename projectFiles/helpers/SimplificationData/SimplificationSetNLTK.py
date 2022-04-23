from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.seq2seq.constants import EOS, indices


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

    def addIndices(self):
        self.originalIndices = [indices[word] for word in self.originalTokenized] + [EOS]
        self.allSimpleIndices = [[indices[word] for word in sentence] + [EOS] for sentence in self.allSimpleTokenized]
