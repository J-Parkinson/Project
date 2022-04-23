from projectFiles.helpers.DatasetToLoad import dsName
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.initialiseCurriculumLearning import initialiseCurriculumLearning
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN
from projectFiles.seq2seq.training import trainMultipleIterations


def runSeq2Seq(dataset, embedding, hiddenLayerWidthForIndices=512, maxIndices=253401, curriculumLearningMD=None):
    hiddenLayerWidth = getHiddenSize(hiddenLayerWidthForIndices, embedding)
    datasetLoaded = simplificationDataToPyTorch(dataset, embedding)
    print("Dataset loaded")
    if curriculumLearningMD:
        datasetLoaded = initialiseCurriculumLearning(datasetLoaded, curriculumLearningMD)

    # batching
    datasetBatches = simplificationDatasetLoader(datasetLoaded)

    # 0=SOS, 1=EOS, 2=0', etc.
    print("Creating encoder and decoder")
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth, embedding).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)
    print("Begin iterations")
    epochData = trainMultipleIterations(encoder=encoder, decoder=decoder, allData=datasetLoaded,
                                        datasetName=dsName(dataset), embedding=embedding)
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
