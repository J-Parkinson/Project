from projectFiles.helpers.DatasetToLoad import dsName
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.helpers.getMaxLens import getMaxLens
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.initialiseCurriculumLearning import initialiseCurriculumLearning
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN
from projectFiles.seq2seq.training import trainMultipleIterations


def runSeq2Seq(dataset, embedding, curriculumLearningMD, hiddenLayerWidthForIndices=256, restrict=200000000,
               batchSize=128, batchesBetweenValidation=50):
    hiddenLayerWidth = getHiddenSize(hiddenLayerWidthForIndices, embedding)

    maxLenSentence = getMaxLens(dataset, embedding, restrict=restrict)

    # Also restricts length of max len sentence in each set (1-n and 1-1)
    datasetLoaded = simplificationDataToPyTorch(dataset, embedding, maxLen=maxLenSentence)
    print("Dataset loaded")

    initialiseCurriculumLearning(datasetLoaded.train, curriculumLearningMD)

    # batching
    datasetBatches = simplificationDatasetLoader(datasetLoaded, embedding, batch_size=batchSize)

    print("Creating encoder and decoder")
    encoder = EncoderRNN(len(indicesReverseList), hiddenLayerWidth, embedding, batchSize).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, len(indicesReverseList), embedding, batchSize, dropout=0.3,
                             max_length=maxLenSentence).to(device)

    print("Begin iterations")
    epochData = trainMultipleIterations(encoder=encoder, decoder=decoder, allData=datasetBatches,
                                        datasetName=dsName(dataset), embedding=embedding, batchSize=batchSize,
                                        hiddenLayerWidth=hiddenLayerWidth, curriculumLearning=curriculumLearningMD.flag,
                                        maxLenSentence=maxLenSentence,
                                        noTokens=len(indicesReverseList),
                                        batchesBetweenValidation=batchesBetweenValidation)
    return epochData