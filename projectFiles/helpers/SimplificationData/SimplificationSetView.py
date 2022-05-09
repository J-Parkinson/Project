# We created views to allow us to deal with individual pairs easier than in a set
# they are not designed to be turned back into sets or offer any methods -- we instead operate on sets

class simplificationSetView():
    def __init__(self, original, simple, originalTokenized, simpleTokenized, originalTokenizedPadded,
                 simpleTokenizedPadded, dataset, language, predicted, originalIndices, simpleIndices, originalTorch,
                 simpleTorch, maxSentenceLen):
        self.original = original
        self.simple = simple
        self.originalTokenized = originalTokenized
        self.simpleTokenized = simpleTokenized
        self.originalTokenizedPadded = originalTokenizedPadded
        self.simpleTokenizedPadded = simpleTokenizedPadded
        self.dataset = dataset
        self.language = language
        self.predicted = predicted
        self.originalIndices = originalIndices
        self.simpleIndices = simpleIndices
        self.originalTorch = originalTorch
        self.simpleTorch = simpleTorch
        self.maxSentenceLen = maxSentenceLen

    def addPredicted(self, prediction):
        self.predicted = prediction
