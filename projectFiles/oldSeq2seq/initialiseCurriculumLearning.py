def initialiseCurriculumLearning(dataset, curriculumLearningMD):
    dataset.train.initialiseCurriculumLearning(curriculumLearningMD.flag,
                                               curriculumLearningMD.lambdaFunc,
                                               curriculumLearningMD.addEpoch)
    return dataset
