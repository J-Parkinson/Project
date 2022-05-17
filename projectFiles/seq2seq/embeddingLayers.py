import torch

from projectFiles.constants import device
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.gloveEmbeddings.loadGloveEmbeddings import gloveEmbeddings


# INPUT EMBEDDING HANDLES [EOS]

def inputEmbeddingLayer(embedding, inputSize, hiddenSize):
    if embedding == embeddingType.indices and inputSize and hiddenSize:
        # Trained embeddings within encoder instead
        return lambda x: x.originalTorch
    else:
        return embeddingGlove


def embeddingGlove(view):
    originalTokenizedEOS = view.originalTokenized + ["[EOS]"]
    embeddings = torch.tensor(gloveEmbeddings(originalTokenizedEOS), dtype=torch.float32,
                              device=device)
    return embeddings
