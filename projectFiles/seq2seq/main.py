from projectFiles.curriculumLearningFunctions.sentenceScoreMetrics import fleschKincaidInput
from projectFiles.evaluation.plotSingleResults import printPlots
from projectFiles.evaluation.saveResults import savePlotData
from projectFiles.helpers.DatasetToLoad import datasetToLoad, dsName
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.runSeq2Seq import runSeq2Seq


def runE2E(dataset, embedding, curriculumLearningSpec):
    print("E2E begins")

    noEpochsForDS = {datasetToLoad.asset: 15, datasetToLoad.wikilarge: 4, datasetToLoad.wikismall: 7}
    datasetName = dsName(dataset)
    locationToSaveToFromProjectFiles = "seq2seq/trainedModels/"

    paramsSameForEveryRun = {
        "batchesBetweenValidationCheck": 50,
        "batchesBetweenTrainingPlots": 20,
        "batchSize": 64,
        "curriculumLearningAfterFirstEpoch": curriculumLearningMetadata(curriculumLearningFlag.randomized),
        "decoderNoLayers": 2,
        "encoderNoLayers": 2,
        "dropout": 0.1,
        "gradientClip": 50,
        "hiddenLayerSize": 512,
        "learningRate": 0.0001,
        "learningRateDecoderMultiplier": 5,
        "locationToSaveToFromProjectFiles": locationToSaveToFromProjectFiles,
        "minOccurencesOfToken": 2,
        "noEpochs": noEpochsForDS[dataset],
        "restrictLengthOfSentences": None,
        "teacherForcingRatio": 1
    }

    print(f"No epochs: {noEpochsForDS[dataset]}")

    batchSize, datasetBatches, decoder, decoderNoLayers, encoder, fileSaveDir, iterationGlobal, plotDevLosses, \
    plotLosses, resultsGlobal = runSeq2Seq(dataset, datasetName, embedding, curriculumLearningSpec,
                                           **paramsSameForEveryRun)
    printPlots(resultsGlobal, plotLosses, plotDevLosses, datasetName, embedding, fileSaveDir,
               len(datasetBatches.trainDL), noEpochsForDS[dataset])
    savePlotData(resultsGlobal, plotLosses, plotDevLosses, fileSaveDir)

    # 1. printPlots
    # 2. Save all evaluated test data and plot data
    # 3. calculate EASSE data and save/return
    # 4. Save encoder/decoder


# To get all results I will need, I need to run all of these

# Iter 1: run on each of (indices/glove) and (asset/wikismall/wikilarge)
# runE2E(datasetToLoad.asset, embeddingType.indices,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.wikismall, embeddingType.indices,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.wikilarge, embeddingType.indices,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.asset, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.wikismall, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.randomized))


# Graphs 1: curricLearning sorting func
# Baseline model

# Different sorting funcs
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=numberOfComplexWordInInput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=noTokensInInput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInLengthOfInputAndOutput))
# runE2E(datasetToLoad.wikilarge, embeddingType.glove,
#       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=differenceInFK))
runE2E(datasetToLoad.wikilarge, embeddingType.glove,
       curriculumLearningMetadata(curriculumLearningFlag.orderedCL, lambdaFunc=fleschKincaidInput))
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
