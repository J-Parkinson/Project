from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.evaluate import evaluate
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq

BESTONE = None


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


runE2E(datasetToLoad.asset, embeddingType.indices,
       curriculumLearningMetadata(curriculumLearningFlag.randomized), batchesBetweenValidation=75, restrict=40)

# To get all results I will need, I need to run all of these

# Graphs 1: curricLearning sorting func
# Baseline model
runE2E(datasetToLoad.wikilarge, embeddingType.bert,
       curriculumLearningMetadata(curriculumLearningFlag.randomized), batchesBetweenValidation=75)
# Different sorting funcs
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=numberOfComplexWordInInput), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=noTokensInInput), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInLengthOfInputAndOutput), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInFK), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=fleschKincaidInput), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=bertScore), batchesBetweenValidation=75)
#
#
##Different curriculum schedules (inc. active learning)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.sampledFlatCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.sampledPriorityCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
#
#
##impact on simpler embeddings
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
# runE2E(datasetToLoad.wikilarge, embeddingType.indices,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
#
##impact on smaller datasets
# runE2E(datasetToLoad.asset, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized), batchesBetweenValidation=75)
# runE2E(datasetToLoad.asset, embeddingType.bert,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE), batchesBetweenValidation=75)
