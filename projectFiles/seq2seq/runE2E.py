from projectFiles.curriculumLearningFunctions.length import noTokensInInput
from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.seq2seq.evaluate import evaluateAllEpoch
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq


def runE2E(dataset, curriculumLearningMD=None):
    epochData = runSeq2Seq(dataset, curriculumLearningMD=curriculumLearningMD)
    epochData.printPlots()
    epochData.savePlotData()
    epochData = evaluateAllEpoch(epochData)
    epochData.saveTestData()
    epochData.evaluateEASSE()

    # 1. printPlots
    # 2. Save all evaluated test data and plot data
    # 3. calculate EASSE data and save/return
    # 4. Save encoder/decoder


runE2E(datasetToLoad.wikilarge)
runE2E(datasetToLoad.wikilarge,
       curriculumLearningMD=curriculumLearningMetadata(curriculumLearningFlag.full, noTokensInInput))
