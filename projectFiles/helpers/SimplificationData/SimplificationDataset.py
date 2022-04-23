from random import shuffle, random

from torch.utils.data import Dataset

from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag


# Torch dataset object, handles curriculum learning for the sentences contained within it, handles fetching sentences

class simplificationDataset(Dataset):

    def __init__(self, simplificationPairSet):
        self.dataset = simplificationPairSet
        shuffle(self.dataset)
        self.curriculumLearning = curriculumLearningFlag.off
        self.curriculumLearningIndices = []
        self.curriculumLearningPercentageAddedPerEpoch = 0
        self.curriculumLearningPercentage = 0
        self.lambdaFunc = None
        self.epochNo = 1

    # Active-learning-esque accumulation of epochs (n% at first, then 2n%, etc.)

    def _distributionForCurriculumLearningEpochFlat(self):
        location = random()
        # If initial epoch or entire dataset ran through
        if self.epochNo == 1 or (self.epochNo * self.curriculumLearningPercentageAddedPerEpoch) > 100:
            return int(location * len(self.dataset))
        else:
            return int(location * len(self))

    # Active-learning-esque accumulation of epochs (n% at first, then 2n%, etc.) with emphasis on new data introduced
    def _distributionForCurriculumLearningEpoch(self):
        location = random()
        # If initial epoch or entire dataset ran through
        if self.epochNo == 1 or (self.epochNo * self.curriculumLearningPercentageAddedPerEpoch) > 100:
            return int(location * len(self.curriculumLearningIndices))
        else:
            threshold = int(len(self) * (
                    self.curriculumLearningPercentage - self.curriculumLearningPercentageAddedPerEpoch) // self.curriculumLearningPercentage)
            if location < 0.5:
                return int(location * 2 * threshold)
            else:
                maxValue = len(self)
                factor = maxValue - threshold
                return int(threshold + factor * ((location - 0.5) * 2))

    #Every dataset is initialised without curriculum learning, and it is initialised later (since we also need to provide a curriculum learning function)
    def initialiseCurriculumLearning(self, flag, lambdaFunc, addEpoch=10):
        self.curriculumLearning = flag
        self.cLPercentAddedPerEpoch = addEpoch
        self.lambdaFunc = lambdaFunc
        self.curriculumLearningPercentage = addEpoch
        self._curriculumLearningSortDataset(self.lambdaFunc)

    # For increasing percentage of dataset returned for full, epoch and epochFlat
    def increasePercentage(self):
        self.epochNo += 1
        self.curriculumLearningPercentage = min(100,
                                                self.curriculumLearningPercentageAddedPerEpoch + self.curriculumLearningPercentage)

    #Override default len() behaviour
    def __len__(self):
        if self.curriculumLearning.value >= curriculumLearningFlag.full.value:
            return len(self.dataset) * self.curriculumLearningPercentage // 100
        return len(self.dataset)

    #Override default [] notation behaviour to enable curriculum learning using our custom dataset object
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[idxV] for idxV in range(idx.start or 0, idx.stop or len(self), idx.step or 1)]
        elif isinstance(idx, int):
            if idx >= self.__len__():
                raise IndexError(
                    f"list index out of range{' due to curriculum learning' if len(self.dataset) > idx >= len(self) else ''}")
            if self.curriculumLearning == curriculumLearningFlag.epoch:
                (xIndex, yIndex) = self.curriculumLearningIndices[self._distributionForCurriculumLearningEpoch()]
                return self.dataset[xIndex].getView(yIndex)
            elif self.curriculumLearning == curriculumLearningFlag.epochFlat:
                (xIndex, yIndex) = self.curriculumLearningIndices[self._distributionForCurriculumLearningEpochFlat()]
                return self.dataset[xIndex].getView(yIndex)
            elif self.curriculumLearning.value:
                (xIndex, yIndex) = self.curriculumLearningIndices[idx]
                return self.dataset[xIndex].getView(yIndex)
            return self.dataset[idx]
        elif isinstance(idx, tuple):
            raise TypeError('list indices must be integers or slices, not tuple')
        else:
            raise TypeError(f'invalid argument type: {type(idx)}')

    # First order function which takes curriculum learning function and applies this to the dataset
    # Returned indices are then used to fetch each element of the dataset (since otherwise we would be unable to
    # change the lambda func mid training
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
