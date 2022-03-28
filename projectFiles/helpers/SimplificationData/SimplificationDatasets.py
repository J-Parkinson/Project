from projectFiles.helpers.DatasetSplits import datasetSplits


class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def addIndices(self, indices, maxIndices=75000):
        for set in self.train.dataset:
            set.addIndices(indices, maxIndices)
        for set in self.dev.dataset:
            set.addIndices(indices, maxIndices)
        for set in self.test.dataset:
            set.addIndices(indices, maxIndices)

    def torchProcess(self):
        dict = {}
        dict["train"] = [set.torchSet() for set in self.train.dataset]
        dict["dev"] = [set.torchSet() for set in self.train.dataset]
        dict["test"] = [set.torchSet() for set in self.train.dataset]
        return dict
