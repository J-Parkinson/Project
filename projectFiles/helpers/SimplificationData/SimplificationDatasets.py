class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def addIndices(self, indices, maxIndices=222823):
        for set in self.train.dataset:
            set.addIndices(indices, maxIndices)
        for set in self.dev.dataset:
            set.addIndices(indices, maxIndices)
        for set in self.test.dataset:
            set.addIndices(indices, maxIndices)

    def torchProcess(self):
        for dataset in [self.train.dataset, self.dev.dataset, self.test.dataset]:
            for set in dataset:
                set.torchSet()
        return
