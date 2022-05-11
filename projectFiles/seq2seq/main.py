from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.evaluate import evaluate
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq


def runE2E(dataset, embedding, curriculumLearningMD, restrict=200000000, batchesBetweenValidation=50):
    epochData = runSeq2Seq(dataset, embedding, curriculumLearningMD, restrict=restrict,
                           batchesBetweenValidation=batchesBetweenValidation)
    epochData.printPlots()
    epochData.savePlotData()
    allData = evaluate(epochData)
    epochData.saveTestData(allData)
    epochData.evaluateEASSE(allData)

    # 1. printPlots
    # 2. Save all evaluated test data and plot data
    # 3. calculate EASSE data and save/return
    # 4. Save encoder/decoder


runE2E(datasetToLoad.asset, embeddingType.glove,
       curriculumLearningMetadata(curriculumLearningFlag.randomized), batchesBetweenValidation=15)
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMD=curriculumLearningMetadata(curriculumLearningFlag.orderedCL, noTokensInInput))
# runE2E(datasetToLoad.asset, embeddingType.bert,
#       curriculumLearningMD=curriculumLearningMetadata(curriculumLearningFlag.orderedCL, noTokensInInput))
