from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch


def checkCurriculumLearningWorks():
    asset = simplificationDataToPyTorch(datasetToLoad.asset)
