import torch

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model as BERTmodel
from projectFiles.preprocessing.gloveEmbeddings.loadGloveEmbeddings import gloveEmbeddings
from projectFiles.seq2seq.constants import device, bertWidth


# INPUT EMBEDDING HANDLES [EOS]

def inputEmbeddingLayer(embedding, inputSize, hiddenSize):
    if embedding == embeddingType.indices and inputSize and hiddenSize:
        # Trained embeddings within encoder instead
        return lambda x: x.originalTorch
    elif embedding == embeddingType.glove:
        return embeddingGlove
    else:
        return embeddingBert


def embeddingGlove(view):
    originalTokenizedEOS = view.originalTokenized + ["[EOS]"]
    embeddings = torch.tensor(gloveEmbeddings(originalTokenizedEOS), dtype=torch.float32,
                              device=device)
    return embeddings


def embeddingBert(view):
    segmentIDs = [1] * len(view.originalTorch)
    segmentTorch = torch.tensor([segmentIDs], dtype=torch.int32, device=device)
    tokensTorch = torch.tensor(view.originalTorch.T, dtype=torch.int32, device=device)
    with torch.no_grad():
        embeddingsUnformatted = BERTmodel(tokensTorch, segmentTorch)
        embeddingsHiddenLayer = torch.stack(list(embeddingsUnformatted.hidden_states), dim=0).squeeze()
        # 12 BERT layers + initial input
        # https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca suggests using the sum of the last four layers
        EOSBert = torch.tensor([1 for _ in range(bertWidth)], dtype=torch.int32, device=device)
        embeddingsNEOS = torch.sum(embeddingsHiddenLayer[-4:, 1:-1], dim=0)
        embeddings = torch.cat([embeddingsNEOS, EOSBert.reshape(1, -1)], dim=0)

    return embeddings
