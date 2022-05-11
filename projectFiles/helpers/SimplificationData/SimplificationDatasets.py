from projectFiles.helpers.SimplificationData.setToBERTNLTK import convertSetForEmbeddingAndPadding


# Stores training, dev and test set splits
class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    #Loads existing dataset and applies embedding type to each (i.e. tokenizing and creating Torch objects from each)
    def loadFromPickleAndPad(self, embedding, maxLenSentence):
        totalLen = len(self.train.dataset) + len(self.dev.dataset) + len(self.test.dataset)
        nPercent = [totalLen * n // 100 for n in range(100)]
        x = 0
        # self.train[0] fails at the moment due to no indices being created
        # this is done in the curriculum learning initialisation phase
        for set in self.train.dataset:
            convertSetForEmbeddingAndPadding(set, embedding, maxLenSentence)
            if x in nPercent:
                print(f"{nPercent.index(x)}% complete")
            x += 1
        for set in self.dev.dataset:
            convertSetForEmbeddingAndPadding(set, embedding, maxLenSentence)
            if x in nPercent:
                print(f"{nPercent.index(x)}% complete")
            x += 1
        for set in self.test.dataset:
            convertSetForEmbeddingAndPadding(set, embedding, maxLenSentence)
            if x in nPercent:
                print(f"{nPercent.index(x)}% complete")
            x += 1

    def addIndices(self):
        for set in self.train:
            set.addIndices()
        for set in self.dev:
            set.addIndices()
        for set in self.test:
            set.addIndices()

    def torchProcess(self):
        for dataset in [self.train, self.dev, self.test]:
            for set in dataset:
                set.torchSet()
        return

    def filterOutLongSentence(self, maxLen):
        newTrain = []
        for set in self.train.dataset:
            score = set.removeByLength(maxLen)
            if score:
                newTrain.append(set)
        self.train.dataset = newTrain

        newDev = []
        for set in self.dev.dataset:
            score = set.removeByLength(maxLen)
            if score:
                newDev.append(set)
        self.dev.dataset = newDev

        newTest = []
        for set in self.test.dataset:
            score = set.removeByLength(maxLen)
            if score:
                newTest.append(set)
        self.test.dataset = newTest
