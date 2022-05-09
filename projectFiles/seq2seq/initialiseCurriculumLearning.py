from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag


def initialiseCurriculumLearning(dataset, curriculumLearningMD):
    dataset.train.initialiseCurriculumLearning(curriculumLearningMD.flag,
                                               curriculumLearningMD.lambdaFunc)
    dataset.dev.initialiseCurriculumLearning(curriculumLearningFlag.noCL)
    dataset.test.initialiseCurriculumLearning(curriculumLearningFlag.noCL)
    return dataset
