from enum import Enum


class curriculumLearningFlag(Enum):
    off = 0
    basic = 1
    full = 2
    epoch = 3
    epochFlat = 4


class curriculumLearningMetadata():
    def __init__(self, flag, lambdaFunc, addEpoch=10):
        self.flag = flag
        self.lambdaFunc = lambdaFunc
        self.addEpoch = addEpoch

# Basic -> orders training data, feeds in all at once every epoch
# Full -> orders training data, feeds in n% at first, then 2n%, etc. until 100% is fed in.
