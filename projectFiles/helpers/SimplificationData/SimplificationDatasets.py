class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def addIndices(self, indices, maxIndices=222823):
        for set in self.train:
            set.addIndices(indices, maxIndices)
        for set in self.dev:
            set.addIndices(indices, maxIndices)
        for set in self.test:
            set.addIndices(indices, maxIndices)

    def torchProcess(self):
        for dataset in [self.train, self.dev, self.test]:
            for set in dataset:
                set.torchSet()
        return
