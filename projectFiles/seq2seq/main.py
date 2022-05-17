from projectFiles.evaluation.saveResults import savePlotData

print("Running main - loading imports")
from projectFiles.evaluation.plots import printPlots
from projectFiles.helpers.DatasetToLoad import datasetToLoad, dsName
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq

print("Imports loaded")


def runE2E(dataset, embedding, curriculumLearningSpec):
    print("E2E begins")

    noEpochsForDS = {datasetToLoad.asset: 2, datasetToLoad.wikilarge: 2, datasetToLoad.wikismall: 4}
    datasetName = dsName(dataset)
    locationToSaveToFromProjectFiles = "seq2seq/trainedModels/"

    paramsSameForEveryRun = {
        "batchesBetweenValidationCheck": 75 if embedding == embeddingType.indices else 50,
        "batchesBetweenTrainingPlots": 20,
        "batchSize": 64,
        "curriculumLearningAfterFirstEpoch": curriculumLearningMetadata(curriculumLearningFlag.randomized),
        "dropout": 0.1,
        "gradientClip": 50,
        "hiddenLayerSize": 512,
        "learningRate": 0.0001,
        "learningRateDecoderMultiplier": 5,
        "locationToSaveToFromProjectFiles": locationToSaveToFromProjectFiles,
        "minOccurencesOfToken": 2,
        "noEpochs": noEpochsForDS[dataset],
        "noLayersDecoder": 2,
        "noLayersEncoder": 2,
        "restrictLengthOfSentences": None,
        "teacherForcingRatio": 1
    }

    print(f"No epochs: {noEpochsForDS[dataset]}")

    datasetBatches, decoder, encoder, iterationGlobal, plotDevLosses, plotLosses, resultsGlobal, fileSaveDir = \
        runSeq2Seq(dataset, datasetName, embedding, curriculumLearningSpec, **paramsSameForEveryRun)
    printPlots(resultsGlobal, plotLosses, plotDevLosses, datasetName, embedding, fileSaveDir,
               len(datasetBatches.trainDL), noEpochsForDS[dataset])
    savePlotData(resultsGlobal, plotLosses, plotDevLosses, fileSaveDir)
    # allData = validationEvaluationLoss(epochData, datasetSplits.test)
    # epochData.saveTestData(allData)
    # epochData.evaluateEASSE(allData)

    # 1. printPlots
    # 2. Save all evaluated test data and plot data
    # 3. calculate EASSE data and save/return
    # 4. Save encoder/decoder


# runE2E(datasetToLoad.asset, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized), batchesBetweenValidation=75, restrict=40)

# To get all results I will need, I need to run all of these

# Graphs 1: curricLearning sorting func
# Baseline model
runE2E(datasetToLoad.asset, embeddingType.indices,
       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# Different sorting funcs
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=numberOfComplexWordInInput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=noTokensInInput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInLengthOfInputAndOutput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInFK))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=fleschKincaidInput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=bertScore))
#
#
##Different curriculum schedules (inc. active learning)
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.sampledFlatCL, lambdaFunc=BESTONE))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.sampledPriorityCL, lambdaFunc=BESTONE))
#
#
##impact on simpler embeddings
# runE2E(datasetToLoad.wikilarge, embeddingType.indices,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE))
#
##impact on smaller datasets
# runE2E(datasetToLoad.asset, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.asset, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=BESTONE))
