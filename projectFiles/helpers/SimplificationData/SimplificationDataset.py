from random import shuffle

from torch.utils.data import Dataset

from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag


class simplificationDataset(Dataset):

    def __init__(self, simplificationPairSet):
        self.dataset = simplificationPairSet
        shuffle(self.dataset)
        self.curriculumLearning = curriculumLearningFlag.off
        self.curriculumLearningIndices = []
        self.curriculumLearningPercentageAddedPerEpoch = 0
        self.curriculumLearningPercentage = 0
        self.lambdaFunc = None

    def initialiseCurriculumLearning(self, flag, lambdaFunc, addEpoch=10):
        self.curriculumLearning = flag
        self.cLPercentAddedPerEpoch = addEpoch
        self.lambdaFunc = lambdaFunc
        self.curriculumLearningPercentage = addEpoch
        self._curriculumLearningSortDataset(self.lambdaFunc)

    def increasePercentage(self):
        self.curriculumLearningPercentage = min(100,
                                                self.curriculumLearningPercentageAddedPerEpoch + self.curriculumLearningPercentage)

    def __len__(self):
        if self.curriculumLearning == curriculumLearningFlag.full:
            return len(self.dataset) * self.curriculumLearningPercentage // 100
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[idxV] for idxV in range(*idx.indices(3))]
        elif isinstance(idx, int):
            if idx >= self.__len__():
                raise IndexError(
                    f"list index out of range{' due to curriculum learning' if len(self.dataset) > idx >= len(self) else ''}")
            if self.curriculumLearning.value:
                (xIndex, yIndex) = self.curriculumLearningIndices[idx]
                return (self.dataset[xIndex], yIndex)
            return self.dataset[idx]
        elif isinstance(idx, tuple):
            raise TypeError('list indices must be integers or slices, not tuple')
        else:
            raise TypeError(f'invalid argument type: {type(idx)}')

    def _curriculumLearningSortDataset(self, lambdaFunc):
        metricOverDataset = [setVal.getMetric(lambdaFunc) for setVal in self.dataset]
        metricWithIndicesOverDataset = [[(metricValue, (i, j)) for j, metricValue in enumerate(currentSet)] for
                                        i, currentSet in enumerate(metricOverDataset)]
        metricWithIndicesOverDatasetFlattened = [metricValueIndex for set in metricWithIndicesOverDataset for
                                                 metricValueIndex in set]
        # Sort is stable, items remain in order if metric same for two values
        # Will always go smallest to largest, use negatives if largest to smallest
        metricWithIndicesOverDatasetFlattened.sort(key=lambda x: x[0])
        self.curriculumLearningIndices = [y for (x, y) in metricWithIndicesOverDatasetFlattened]
