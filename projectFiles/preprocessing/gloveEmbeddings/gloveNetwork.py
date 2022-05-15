import torch
from torch import nn

from projectFiles.preprocessing.gloveEmbeddings.loadGloveEmbeddings import gloveModel
from projectFiles.seq2seq.constants import gloveWidth, device


def getGloveEmbeddingNN(noTokens, embeddingSize=gloveWidth):
    embeddingLayer = nn.Embedding(noTokens, embeddingSize)
    embeddingMatrix = gloveModel.vectors
    embeddingLayer.weight = nn.Parameter(torch.tensor(embeddingMatrix, dtype=torch.float32, device=device))
    embeddingLayer.requires_grad_(False)
    return embeddingLayer


class GloveEmbeddings(nn.Module):
    def __init__(self, noTokens, embeddingSize=gloveWidth):
        super(GloveEmbeddings, self).__init__()
        self.noTokens = noTokens
        self.embeddingSize = embeddingSize
        self.embeddingLayer = getGloveEmbeddingNN(noTokens, embeddingSize)

    def forward(self, input):
        return self.embeddingLayer(input)
