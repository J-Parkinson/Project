from random import random

from torch.utils.data import Dataset

from projectFiles.helpers.clamp import clamp
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag


# Torch dataset object, handles curriculum learning for the sentences contained within it, handles fetching sentences

class simplificationDataset(Dataset):

    def __init__(self, simplificationPairSet):
        self.dataset = simplificationPairSet
        self.curriculumLearning = curriculumLearningFlag.noViews
        self.curriculumLearningInstantiated = False
        self.accesses = 0

    # Every dataset is initialised without curriculum learning, and it is initialised later (since we also need to provide a curriculum learning function)
    def initialiseCurriculumLearning(self, flag, lambdaFunc=None):
        self.curriculumLearningInstantiated = bool(flag.value)
        self.curriculumLearning = flag
        if self.curriculumLearningInstantiated:
            if flag == curriculumLearningFlag.noCL:
                self.lambdaFunc = lambda _1, _2: random()
            else:
                self.lambdaFunc = lambdaFunc
            self._curriculumLearningSortDataset(self.lambdaFunc)

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

    # Helper
    def _sampleIntFlatRange(self, min, max):
        return int(min) + int(random() * (max - min))

    # Active-learning-esque accumulation
    def _sampleFlat(self):
        # If initial epoch or entire dataset ran through
        return self._sampleIntFlatRange(0, clamp(self.accesses, len(self.curriculumLearningIndices) // 20,
                                                 len(self.curriculumLearningIndices)))

    # Active-learning-esque accumulation of epochs (n% at first, then 2n%, etc.) with emphasis on new data introduced
    def _samplePriority(self):
        if self.accesses < len(self.curriculumLearningIndices) // 10:
            return self._sampleFlat()
        threshold = random()
        if threshold > 0.5:
            return self._sampleIntFlatRange(min(self.accesses, len(self.curriculumLearningIndices))
                                            - len(self.curriculumLearningIndices) // 20,
                                            min(self.accesses, len(self.curriculumLearningIndices)))
        else:
            return self._sampleIntFlatRange(0, min(self.accesses, len(self.curriculumLearningIndices))
                                            - len(self.curriculumLearningIndices) // 20)

    # Override default len() behaviour
    def __len__(self):
        if self.curriculumLearningInstantiated:
            return len(self.curriculumLearningIndices)
        return len(self.dataset)

    # Override default [] notation behaviour to enable curriculum learning using our custom dataset object
    def __getitem__(self, idx):
        if not self.curriculumLearningInstantiated:
            raise IndexError("Curriculum learning not instantiated.")
        if isinstance(idx, slice):
            return [self[idxV] for idxV in range(idx.start or 0, idx.stop or len(self), idx.step or 1)]
        elif isinstance(idx, int):
            self.accesses += 1
            if idx >= len(self):
                raise IndexError(
                    f"list index out of range{' due to curriculum learning' if len(self.dataset) > idx >= len(self) else ''}")
            if self.curriculumLearning == curriculumLearningFlag.sampledPriorityCL:
                (xIndex, yIndex) = self.curriculumLearningIndices[self._samplePriority()]
                return self.dataset[xIndex].getView(yIndex)
            elif self.curriculumLearning == curriculumLearningFlag.sampledFlatCL:
                (xIndex, yIndex) = self.curriculumLearningIndices[self._sampleFlat()]
                return self.dataset[xIndex].getView(yIndex)
            elif self.curriculumLearning == curriculumLearningFlag.noViews:
                return self.dataset[idx]
            else:
                (xIndex, yIndex) = self.curriculumLearningIndices[idx]
                return self.dataset[xIndex].getView(yIndex)
        elif isinstance(idx, tuple):
            raise TypeError('list indices must be integers or slices, not tuple')
        else:
            raise TypeError(f'invalid argument type: {type(idx)}')
