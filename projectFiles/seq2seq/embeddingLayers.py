import torch

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model as BERTmodel
from projectFiles.preprocessing.gloveEmbeddings.loadGloveEmbeddings import getGloveEmbeddings
from projectFiles.seq2seq.constants import device, getIndexRaw


def embeddingLayer(embedding, inputSize, hiddenSize):
    if embedding == embeddingType.indices and inputSize and hiddenSize:
        # Trained embeddings within encoder instead
        return lambda x: x
    elif embedding == embeddingType.glove:
        return embeddingGlove
    else:
        return embeddingBert


def embeddingGlove(indices):
    embeddings = torch.tensor([getGloveEmbeddings(getIndexRaw(index - 2)) for index in indices], dtype=torch.float32,
                              device=device)
    return embeddings


def embeddingBert(tokensTorch):
    segmentIDs = [1] * len(tokensTorch)
    segmentTorch = torch.tensor([segmentIDs], dtype=torch.int32, device=device)
    tokensTorch = torch.tensor(tokensTorch.T, dtype=torch.int32, device=device)
    with torch.no_grad():
        embeddingsUnformatted = BERTmodel(tokensTorch, segmentTorch)
        embeddingsHiddenLayer = torch.stack(list(embeddingsUnformatted.hidden_states), dim=0).squeeze()
        # 12 BERT layers + initial input
        # https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca suggests using the sum of the last four layers
        embeddings = torch.sum(embeddingsHiddenLayer[-4:], dim=0)
    return embeddings
