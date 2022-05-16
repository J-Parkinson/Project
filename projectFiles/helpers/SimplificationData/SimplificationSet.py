from unidecode import unidecode

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData.SimplificationSetView import simplificationSetView


class simplificationSet():
    def __init__(self, original, allSimple, dataset, language="en"):
        self.original = original.lower()
        self.allSimple = [sent.lower() for sent in allSimple]
        self._removeEscapedCharacters(dataset)
        self._makeReplacementsGEC()
        self._toAscii()

        self.originalTokenized = self.original
        self.allSimpleTokenized = self.allSimple

        self.dataset = dataset
        self.language = language
        self.predicted = ""

        self.originalIndices = None
        self.allSimpleIndices = None

    # WikiSmall/Large escapes brackets and arrows, which we add back in
    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        for (frm, to) in [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-LCB-", "{"),
                          ("-RCB-", "}"), ("-LAB-", "<"), ("-RAB-", ">")]:
            sentence = sentence.replace(frm, to)
        return sentence

    # Gector (what another text simplification model is based on) suggests replacing these strings in the text
    def _makeReplacementsGEC(self):
        for (frm, to) in [("''", '"'), ("--", "-"), ("`", "'")]:
            self.original = self.original.replace(frm, to)
            self.allSimple = [sentence.replace(frm, to) for sentence in self.allSimple]

    def _toAscii(self):
        self.original = unidecode(self.original.lower().strip())
        self.allSimple = [unidecode(sent.lower().strip()) for sent in self.allSimple]

    # Run before tokenisation
    def _removeEscapedCharacters(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self._removeEscapedBracketsFromSentenceWiki(self.original)
            self.allSimple = [self._removeEscapedBracketsFromSentenceWiki(sentence) for sentence in
                              self.allSimple]
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
                                     None if not self.allSimpleIndices else self.allSimpleIndices[yIndex])

    def getAllViews(self):
        return [self.getView(yIndex) for yIndex in range(len(self.allSimple))]
