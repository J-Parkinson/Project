from projectFiles.helpers.SimplificationData.setToNLTK import convertSetForEmbeddingAndPaddingAndFlagLong

# Stores training, dev and test set splits
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import reinitialiseEmbeddings, indicesReverseList


class simplificationDatasets():
    def __init__(self, dataset, train, dev, test):
        self.dataset = dataset
        self.train = train
        self.dev = dev
        self.test = test

    def reIndex(self):
        reinitialiseEmbeddings()

        print("Performing indices calculations again")
        # self.train[0] fails at the moment due to no indices being created
        # this is done in the curriculum learning initialisation phase
        print("Processing training split...")
        for setA in self.train.dataset:
            setA.addIndices()
        print("Processing dev split...")
        for setA in self.dev.dataset:
            setA.addIndices()
        print("Processing test split...")
        for setA in self.test.dataset:
            setA.addIndices()

    # Loads existing dataset and applies embedding type to each (i.e. tokenizing and creating Torch objects from each)
    def loadFromPickleAndPadAndDeleteLongUncommon(self, embedding, maxLenSentence, minOccurences):
        print("Loading from Pickle - this may take a while.")
        # self.train[0] fails at the moment due to no indices being created
        # this is done in the curriculum learning initialisation phase
        print("Processing training split...")
        tooLongTrain = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence, minOccurences) for
                        setA in
                        self.train.dataset]
        print("Processing dev split...")
        tooLongDev = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence, minOccurences) for
                      setA in
                      self.dev.dataset]
        print("Processing test split...")
        tooLongTest = [convertSetForEmbeddingAndPaddingAndFlagLong(setA, embedding, maxLenSentence, minOccurences) for
                       setA in
                       self.test.dataset]
        print("Filtering splits on length and uncommon tokens...")
        self.train.dataset = [x for x in tooLongTrain if x is not None]
        self.dev.dataset = [x for x in tooLongDev if x is not None]
        self.test.dataset = [x for x in tooLongTest if x is not None]
        print(f"Original/cut length of train split: {len(tooLongTrain)}->{len(self.train.dataset)}")
        print(f"Original/cut length of dev split: {len(tooLongDev)}->{len(self.dev.dataset)}")
        print(f"Original/cut length of test split: {len(tooLongTest)}->{len(self.test.dataset)}")
        print(f"Original number of tokens: {len(indicesReverseList)}")

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