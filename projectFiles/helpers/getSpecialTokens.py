import torch

from projectFiles.constants import device
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS


def getDecoderInput(size):
    return torch.tensor([[SOS for _ in range(size)]], dtype=torch.long, device=device)
