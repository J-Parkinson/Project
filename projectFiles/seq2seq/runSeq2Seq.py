import torch
from torch import optim

from projectFiles.constants import projectLoc, device
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.epochTiming import Timer
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.helpers.getMaxLens import getMaxLens
from projectFiles.helpers.makeDir import makeDir
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.decoderModel import AttnDecoderRNN
from projectFiles.seq2seq.encoderModel import EncoderRNN
from projectFiles.seq2seq.trainingLoops import trainMultipleEpochsWithEvaluation


def runSeq2Seq(dataset, datasetName, embedding, curriculumLearningSpec, hiddenLayerSize, restrictLengthOfSentences,
               minOccurencesOfToken, batchSize, decoderNoLayers, encoderNoLayers, dropout,
               locationToSaveToFromProjectFiles, learningRate, learningRateDecoderMultiplier, **paramsSameForEveryRun):
    hiddenSize = getHiddenSize(embedding, hiddenLayerSize)

    maxLenSentence = getMaxLens(dataset, restrict=restrictLengthOfSentences)

    # Also restricts length of max len sentence in each set (1-n and 1-1)
    datasetLoaded = simplificationDataToPyTorch(dataset, embedding, curriculumLearningSpec, maxLen=maxLenSentence,
                                                minOccurences=minOccurencesOfToken)
    print("Dataset loaded")

    # batching
    datasetBatches = simplificationDatasetLoader(datasetLoaded, batchSize=batchSize)

    print("Creating encoder and decoder")

    embeddingTokenSize = len(indicesReverseList)

    print(f"No indices: {embeddingTokenSize}")

    encoder = EncoderRNN(hiddenSize, embeddingTokenSize, embedding, noLayers=encoderNoLayers, dropout=dropout).to(
        device)
    decoder = AttnDecoderRNN(hiddenSize, embeddingTokenSize, embedding, noLayers=decoderNoLayers, dropout=dropout,
                             maxLength=maxLenSentence).to(device)

    timer = Timer()

    if curriculumLearningSpec.lambdaFunc:
        addCurricFuncName = f'{curriculumLearningSpec.lambdaFunc.__name__}_'
    else:
        addCurricFuncName = ""

    fileSaveDir = makeDir(f"{locationToSaveToFromProjectFiles}{datasetName}_CL-{curriculumLearningSpec.flag.name}"
                          f"_{embedding.name}_{addCurricFuncName}{timer.getStartTime().replace(':', '')}")
    returnFileSaveDir = f"{projectLoc}/{fileSaveDir}"

    encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
    decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate * learningRateDecoderMultiplier)

    for state in encoderOptimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoderOptimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    print("Begin iterations")

    paramsCreatedBeforeTraining = {
        "batches": datasetBatches,
        "batchSize": batchSize,
        "curriculumLearningSpec": curriculumLearningSpec,
        "datasetName": datasetName,
        "datasetLoaded": datasetLoaded,
        "decoder": decoder,
        "decoderNoLayers": decoderNoLayers,
        "decoderOptimizer": decoderOptimizer,
        "encoder": encoder,
        "encoderOptimizer": encoderOptimizer,
        "fileSaveDir": fileSaveDir,
        "timer": timer
    }
    decoder, encoder, iterationGlobal, plotDevLosses, plotLosses, resultsGlobal = \
        trainMultipleEpochsWithEvaluation(**paramsSameForEveryRun, **paramsCreatedBeforeTraining)
    return batchSize, datasetBatches, decoder, decoderNoLayers, encoder, returnFileSaveDir, iterationGlobal, plotDevLosses, \
           plotLosses, resultsGlobal
