import os

import torch

noIndices = 253401
maxLengthSentence = 414
SOS = 0
EOS = 1
teacher_forcing_ratio = 0.5
gloveWidth = 300
bertWidth = 768

fileLoc = "/".join(os.path.abspath(__file__).split("\\")[:-2])

with open(f"{fileLoc}/preprocessing/datasetToIndex/indices.txt", "r", encoding="utf-8") as indicesRead:
    indicesRaw = indicesRead.read()
    indicesRaw = indicesRaw.splitlines()
    indices = {word: i + 2 for i, word in enumerate(indicesRaw)}


def getIndexRaw(index):
    if index >= 2:
        return indicesRaw[index - 2]
    else:
        return ["SOS", "EOS"][index]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")
