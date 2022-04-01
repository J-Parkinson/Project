from projectFiles.helpers.DatasetToLoad import datasetToLoad, dsName
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.loadEncoderDecoder import loadEncoderDecoder, loadDataForEncoderDecoder
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN
from projectFiles.seq2seq.training import trainIters


def runSeq2Seq(dataset, hiddenLayerWidth=256, maxIndices=222823):
    datasetLoaded, datasetOrig = simplificationDataToPyTorch(dataset, maxIndices=maxIndices)
    #0=SOS, 1=EOS, 2=0', etc.
    print("Creating encoder and decoder")
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)
    print("Begin iterations")
    optimalEncoder, optimalDecoder = trainIters(encoder, decoder, datasetLoaded, datasetName=dsName(dataset))
    return optimalEncoder, optimalDecoder, datasetOrig


def runSeq2SeqFromExisting(filepath, hiddenLayerWidth=256, maxIndices=222823):
    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)

    datasetLoaded, datasetOrig, iteration, dataset = loadDataForEncoderDecoder(filepath, maxIndices)
    print("Begin iterations")
    optimalEncoder, optimalDecoder = trainIters(encoder, decoder, datasetLoaded, datasetName=dataset,
                                                startIter=int(iteration))
    return optimalEncoder, optimalDecoder, datasetOrig

runSeq2Seq(
    datasetToLoad.asset
)