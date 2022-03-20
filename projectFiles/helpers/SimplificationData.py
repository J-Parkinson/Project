from projectFiles.helpers.DatasetToLoad import datasetToLoad
from nltk.tokenize import word_tokenize

class simplificationPair:
    def __init__(self, original, simple, dataset, simplicityFactor=(0,4), language="en"):
        self.original = original
        self.simple = simple
        self._removeEscapedCharacters(dataset)
        self._tokenise(dataset)
        self.dataset = dataset
        self.givenSimplicityFactor = simplicityFactor
        self.calculatedSimplicityFactor = 0
        self.language = language

    def _tokenise(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self.original.split(" ")
            self.simple = self.simple.split(" ")
        else:
            self.original = word_tokenize(self.original)
            self.simple = word_tokenize(self.simple)
        self.tokenized = True

    def _removeEscapedBracketsFromSentenceWiki(self, sentence):
        sentence = sentence.replace("-LRB-", "(")
        sentence = sentence.replace("-RRB-", ")")
        sentence = sentence.replace("-LSB-", "[")
        sentence = sentence.replace("-RSB-", "]")
        sentence = sentence.replace("-LCB-", "{")
        sentence = sentence.replace("-RCB-", "}")
        sentence = sentence.replace("-LAB-", "<")
        sentence = sentence.replace("-RAB-", ">")
        return sentence

    #Run before tokenisation
    def _removeEscapedCharacters(self, dataset):
        if dataset in [datasetToLoad.wikilarge, datasetToLoad.wikismall]:
            self.original = self._removeEscapedBracketsFromSentenceWiki(self.original)
            self.simple = self._removeEscapedBracketsFromSentenceWiki(self.simple)
        return

class simplificationDataset:
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test