import torch

from projectFiles.constants import projectLoc
from projectFiles.helpers.DatasetToLoad import name2DTL
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.seq2seq.constants import device
from projectFiles.seq2seq.seq2seqModel import EncoderRNN, AttnDecoderRNN


def loadEncoderDecoder(filepath, hiddenLayerWidth=256, maxIndices=253401):
    encoder = EncoderRNN(maxIndices, hiddenLayerWidth).to(device)
    decoder = AttnDecoderRNN(hiddenLayerWidth, maxIndices, dropout=0.3).to(device)
    encoder.load_state_dict(torch.load(f"{projectLoc}/{filepath}_encoder.pt"))
    decoder.load_state_dict(torch.load(f"{projectLoc}/{filepath}_decoder.pt"))
    print("Encoder and decoder loaded")
    return encoder, decoder


def loadDataForEncoderDecoder(filepath, maxIndices=253401):
    with open(f"{projectLoc}/{filepath}.txt", "r+") as file:
        data = file.readlines()
        [iteration, dataset] = [line.split(" ")[1] for line in data]
        datasetToLoad = name2DTL(dataset)
        print(f"Loading {dataset} dataset")
        datasetData = simplificationDataToPyTorch(datasetToLoad, maxIndices=maxIndices)
    return datasetData, iteration, dataset
