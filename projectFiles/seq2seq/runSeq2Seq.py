from projectFiles.helpers.DatasetToLoad import dsName
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.helpers.getMaxLens import getMaxLens
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.decoderModel import AttnDecoderRNN
from projectFiles.seq2seq.encoderModel import EncoderRNN
from projectFiles.seq2seq.training import trainMultipleIterations


def runSeq2Seq(dataset, embedding, curriculumLearningMD, hiddenLayerWidthForIndices=512, restrict=200000000,
               batchSize=64, batchesBetweenValidation=50, minNoOccurencesForToken=2, encoderNoLayers=2,
               decoderNoLayers=2, dropout=0.1):
    hiddenSize = getHiddenSize(hiddenLayerWidthForIndices, embedding)

    maxLenSentence = getMaxLens(dataset, embedding, restrict=restrict)

    # Also restricts length of max len sentence in each set (1-n and 1-1)
    datasetLoaded = simplificationDataToPyTorch(dataset, embedding, curriculumLearningMD, maxLen=maxLenSentence,
                                                minOccurences=minNoOccurencesForToken)
    print("Dataset loaded")

    # batching
    datasetBatches = simplificationDatasetLoader(datasetLoaded, embedding, batch_size=batchSize)

    print("Creating encoder and decoder")
    print(f"No indices: {len(indicesReverseList)}")
    embeddingTokenSize = len(indicesReverseList)
    encoder = EncoderRNN(embeddingTokenSize, hiddenSize, embedding, noLayers=encoderNoLayers, dropout=dropout).to(
        device)
    decoder = AttnDecoderRNN(hiddenSize, embeddingTokenSize, embedding, noLayers=decoderNoLayers, dropout=dropout,
                             maxLength=maxLenSentence).to(device)

    print("Begin iterations")
    epochData = trainMultipleIterations(encoder=encoder, decoder=decoder, allData=datasetBatches,
                                        datasetName=dsName(dataset), embedding=embedding, batchSize=batchSize,
                                        hiddenLayerWidth=hiddenSize, curriculumLearning=curriculumLearningMD.flag,
                                        maxLenSentence=maxLenSentence,
                                        noTokens=embeddingTokenSize,
                                        batchesBetweenValidation=batchesBetweenValidation)
    return epochData
