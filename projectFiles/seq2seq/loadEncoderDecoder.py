import torch

from projectFiles.helpers.DatasetSplits import datasetSplits
from projectFiles.helpers.DatasetToLoad import name2DTL
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN


def loadEncoderDecoder(filepath, hiddenLayerWidth=256, maxIndices=222823):
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)

    encoder.load_state_dict(torch.load(f"{filepath}_encoder.pt"))
    decoder.load_state_dict(torch.load(f"{filepath}_decoder.pt"))
    print("Encoder and decoder loaded")
    return encoder, decoder

def loadDataForEncoderDecoder(filepath, maxIndices=222823):
    with open(f"{filepath}.txt", "r+") as file:
        data = file.readlines()
        [iteration, dataset] = [line.split(" ")[1] for line in data]
        datasetToLoad = name2DTL(dataset)
        print(f"Loading {dataset} dataset")
        torchObjects, datasetData = simplificationDataToPyTorch(datasetToLoad, startLoc="../../", maxIndices=maxIndices)
    return torchObjects, datasetData, iteration, dataset