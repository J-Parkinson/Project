# We created views to allow us to deal with individual pairs easier than in a set
# they are not designed to be turned back into sets or offer any methods -- we instead operate on sets

class simplificationSetView():
    def __init__(self, original, simple, originalTokenized, simpleTokenized, dataset, language, predicted,
                 originalIndices, simpleIndices):
        self.original = original
        self.simple = simple
        self.originalTokenized = originalTokenized
        self.simpleTokenized = simpleTokenized
        self.dataset = dataset
        self.language = language
        self.predicted = predicted
        self.originalIndices = originalIndices
        self.simpleIndices = simpleIndices

    def addPredicted(self, prediction):
        self.predicted = prediction
