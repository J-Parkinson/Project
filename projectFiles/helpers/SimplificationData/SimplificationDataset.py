from torch.utils.data import Dataset


class simplificationDataset(Dataset):

    def __init__(self, simplificationPairSet):
        self.dataset = simplificationPairSet
        self.curriculumLearning = False
        self.curriculumLearningIndices = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.curriculumLearning:
            (xIndex, yIndex) = self.curriculumLearningIndices[idx]
            return self.dataset[xIndex]
        return self.dataset[idx]

    def curriculumLearningSortDataset(self, lambdaFunc):
        metricOverDataset = [lambdaFunc(setVal) for setVal in self.dataset]
        metricWithIndicesOverDataset = [[(metricValue, (i, j)) for j, metricValue in enumerate(currentSet)] for
                                        i, currentSet in enumerate(metricOverDataset)]
        metricWithIndicesOverDatasetFlattened = [metricValueIndex for set in metricWithIndicesOverDataset for
                                                 metricValueIndex in set]
        # Sort is stable, items remain in order if metric same for two values
        metricWithIndicesOverDatasetFlattened.sort(key=lambda x: lambdaFunc(x[0]))
        self.curriculumLearningIndices = [y for (x, y) in metricWithIndicesOverDatasetFlattened]
        self.curriculumLearning = True
