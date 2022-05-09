from projectFiles.helpers.DatasetToLoad import dsName
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.initialiseCurriculumLearning import initialiseCurriculumLearning
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN
from projectFiles.seq2seq.training import trainMultipleIterations


def runSeq2Seq(dataset, embedding, curriculumLearningMD, hiddenLayerWidthForIndices=256, maxLenSentence=115,
               batchSize=128, earlyStopping=False):
    hiddenLayerWidth = getHiddenSize(hiddenLayerWidthForIndices, embedding)
    print("Creating encoder and decoder")

    # Also restricts length of max len sentence in each set (1-n and 1-1)
    datasetLoaded = simplificationDataToPyTorch(dataset, embedding, maxLen=maxLenSentence)
    print("Dataset loaded")

    if curriculumLearningMD:
        datasetLoaded = initialiseCurriculumLearning(datasetLoaded, curriculumLearningMD)

    # batching
    datasetBatches = simplificationDatasetLoader(datasetLoaded, embedding, batch_size=batchSize,
                                                 shuffle=curriculumLearningMD.flag == curriculumLearningFlag.noCL)

    encoder = EncoderRNN(len(indicesReverseList), hiddenLayerWidth, embedding, batchSize).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, len(indicesReverseList), embedding, batchSize, dropout=0.3,
                             max_length=maxLenSentence).to(device)

    print("Begin iterations")
    epochData = trainMultipleIterations(encoder=encoder, decoder=decoder, allData=datasetBatches,
                                        datasetName=dsName(dataset), embedding=embedding, batchSize=batchSize,
                                        hiddenLayerWidth=hiddenLayerWidth, curriculumLearning=curriculumLearningMD.flag,
                                        earlyStopping=earlyStopping, maxLenSentence=maxLenSentence,
                                        noTokens=len(indicesReverseList))
    return epochData

# def runSeq2SeqFromExisting(filepath, hiddenLayerWidth=256, maxIndices=253401):
#    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)
#
#    datasetLoaded, datasetOrig, iteration, dataset = loadDataForEncoderDecoder(filepath, maxIndices)
#    print("Begin iterations")
#    epochData = trainMultipleIterations(encoder=encoder, decoder=decoder, allData=datasetLoaded, datasetName=dataset,
#                                                startIter=int(iteration))
#    return epochData

# runSeq2Seq(
#    datasetToLoad.asset
# )
