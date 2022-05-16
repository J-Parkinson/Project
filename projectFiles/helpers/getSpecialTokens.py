import torch

from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS
from projectFiles.seq2seq.constants import device


def getDecoderInput(size):
    return torch.tensor([[SOS for _ in range(size)]], dtype=torch.long, device=device)
