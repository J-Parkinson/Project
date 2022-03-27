import torch

from projectFiles.helpers.DatasetToLoad import datasetToLoad, dsName, name2DTL
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN
from projectFiles.seq2seq.training import trainIters


def runSeq2Seq(dataset, hiddenLayerWidth=256, maxIndices=222823):
    datasetLoaded = simplificationDataToPyTorch(dataset, startLoc="../../", maxIndices=maxIndices)
    #0=SOS, 1=EOS, 2=0', etc.
    print("Creating encoder and decoder")
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)
    print("Begin iterations")
    optimalEncoder, optimalDecoder = trainIters(encoder, decoder, datasetLoaded['train'], datasetName=dsName(dataset))


def runSeq2SeqFromExisting(filepath, hiddenLayerWidth=256, maxIndices=222823):
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)

    encoder.load_state_dict(torch.load(f"{filepath}_encoder.pt"))
    decoder.load_state_dict(torch.load(f"{filepath}_decoder.pt"))
    print("Encoder and decoder loaded")

    with open(f"{filepath}.txt", "r+") as file:
        data = file.readlines()
        [iteration, dataset] = [line.split(" ")[1] for line in data]
        datasetToLoad = name2DTL(dataset)
        print(f"Loading {dataset} dataset")
        datasetLoaded = simplificationDataToPyTorch(datasetToLoad, startLoc="../../", maxIndices=maxIndices)
    print("Begin iterations")
    trainIters(encoder, decoder, datasetLoaded['train'], datasetName=dataset, startIter=int(iteration))

def