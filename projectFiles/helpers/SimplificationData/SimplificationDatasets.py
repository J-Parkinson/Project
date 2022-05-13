from projectFiles.helpers.SimplificationData.setToBERTNLTK import convertSetForEmbeddingAndPaddingAndFlagLong


# Stores training, dev and test set splits
class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    # Loads existing dataset and applies embedding type to each (i.e. tokenizing and creating Torch objects from each)
    def loadFromPickleAndPadAndDeleteLong(self, embedding, maxLenSentence):
        print("Loading from Pickle - this may take a while.")
        # self.train[0] fails at the moment due to no indices being created
        # this is done in the curriculum learning initialisation phase
        print("Processing training split...")
        tooLongTrain = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence) for setA in
                        self.train.dataset]
        print("Processing dev split...")
        tooLongDev = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence) for setA in
                      self.dev.dataset]
        print("Processing test split...")
        tooLongTest = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence) for setA in
                       self.test.dataset]
        print("Filtering splits on length...")
        self.train.dataset = [x for x in tooLongTrain if x is not None]
        self.dev.dataset = [x for x in tooLongDev if x is not None]
        self.test.dataset = [x for x in tooLongTest if x is not None]

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