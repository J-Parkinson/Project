class simplificationSetView():
    def __init__(self, original, simple, originalTokenized, simpleTokenized, dataset, language, predicted,
                 originalIndices, simpleIndices, originalTorch, simpleTorch):
        self.original = original
        self.simple = simple
        self.originalTokenized = originalTokenized
        self.simpleTokenized = simpleTokenized
        self.dataset = dataset
        self.language = language
        self.predicted = predicted
        self.originalIndices = originalIndices
        self.simpleIndices = simpleIndices
        self.originalTorch = originalTorch
        self.simpleTorch = simpleTorch
