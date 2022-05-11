from enum import Enum

class curriculumLearningFlag(Enum):
    ordered = 0
    randomized = 1
    orderedCL = 2
    sampledFlatCL = 3
    sampledPriorityCL = 4

class curriculumLearningMetadata():
    def __init__(self, flag, lambdaFunc=None):
        self.flag = flag
        self.lambdaFunc = lambdaFunc

# Basic -> orders training data, feeds in all at once every epoch
# Full -> orders training data, feeds in n% at first, then 2n%, etc. until 100% is fed in.
