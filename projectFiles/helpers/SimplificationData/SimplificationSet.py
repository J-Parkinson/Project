from projectFiles.helpers.DatasetToLoad import datasetToLoad

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

    def addPredicted(self, prediction):
        self.predicted = prediction

    # Returns a list, one for each
    def getMetric(self, lambdaFunc):
        return [lambdaFunc(self.originalTokenized, simplifiedSentence) for
                simplifiedSentence in self.allSimpleTokenized]
