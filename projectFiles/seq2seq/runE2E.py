from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.evaluate import evaluateAllEpoch
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq


def runE2E(dataset, embedding, curriculumLearningMD):
    epochData = runSeq2Seq(dataset, embedding, curriculumLearningMD)
    epochData.printPlots()
    epochData.savePlotData()
    epochData = evaluateAllEpoch(epochData)
    epochData.saveTestData()
    epochData.evaluateEASSE()

    # 1. printPlots
    # 2. Save all evaluated test data and plot data
    # 3. calculate EASSE data and save/return
    # 4. Save encoder/decoder


runE2E(datasetToLoad.asset, embeddingType.bert,
       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMD=curriculumLearningMetadata(curriculumLearningFlag.orderedCL, noTokensInInput))
# runE2E(datasetToLoad.asset, embeddingType.bert,
#       curriculumLearningMD=curriculumLearningMetadata(curriculumLearningFlag.orderedCL, noTokensInInput))
