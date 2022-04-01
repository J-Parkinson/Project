import torch

from projectFiles.seq2seq.constants import EOS, device


def _safeSearch(indices, word, maxIndices=222823):
    try:
        return indices[word.lower()]
    except:
        # indices start at 0
        return maxIndices - 1


def addIndices(sentence, indices, maxIndices=222823):
    return [min(_safeSearch(indices, word, maxIndices - 2), maxIndices - 3) + 2 for word in
            sentence]


def torchSet(indices):
    indices = indices + [EOS]

    indicesTorch = torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    return indicesTorch
