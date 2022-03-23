import torch

maxLengthSentence=383
SOS = 0
EOS = 1
teacher_forcing_ratio = 0.5

with open("../datasetToIndex/indices.txt", "r", encoding="utf-8") as indicesRead:
    indicesRaw = indicesRead.read()
    indicesRaw = indicesRaw.splitlines()
    indices = {word.lower(): i+2 for i, word in enumerate(indicesRaw)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
