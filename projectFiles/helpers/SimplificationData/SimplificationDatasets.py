from projectFiles.helpers.SimplificationData.setToBERTNLTK import convertSetForEmbedding


class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def loadFromPickle(self, embedding):
        totalLen = len(self.train) + len(self.dev) + len(self.test)
        nPercent = [totalLen * n // 100 for n in range(100)]
        x = 0
        for set in self.train:
            convertSetForEmbedding(set, embedding)
            if x in nPercent:
                print(f"{nPercent.index(x)}% complete")
            x += 1
        for set in self.dev:
            convertSetForEmbedding(set, embedding)
            if x in nPercent:
                print(f"{nPercent.index(x)}% complete")
            x += 1
        for set in self.test:
            convertSetForEmbedding(set, embedding)
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
