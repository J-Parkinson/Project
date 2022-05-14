from nltk import word_tokenize

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import getIndex


class simplificationSetNLTK(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self.tokenise()
        self.addIndices()

    def tokenise(self):
        if self.dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.originalTokenized = self.original.split(" ")
            self.allSimpleTokenized = [sentence.split(" ") for sentence in self.allSimple]
        else:
            self.originalTokenized = word_tokenize(self.original)
            self.allSimpleTokenized = [word_tokenize(sentence) for sentence in self.allSimple]
        # ADD SOS/EOS
        self.originalTokenized = self.originalTokenized + ["<eos>"]
        self.allSimpleTokenized = [sentence + ["<eos>"] for sentence in self.allSimpleTokenized]

    def addIndices(self):
        # padding added
        self.originalIndices = [getIndex(word) for word in self.originalTokenized]
        self.allSimpleIndices = [[getIndex(word) for word in sentence] for sentence in
                                 self.allSimpleTokenized]
