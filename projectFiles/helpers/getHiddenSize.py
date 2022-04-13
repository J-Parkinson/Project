from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.constants import gloveWidth, bertWidth


def getHiddenSize(hiddenSize, embedding):
    if embedding == embeddingType.indices:
        return hiddenSize
    elif embedding == embeddingType.glove:
        return gloveWidth
    else:
        return bertWidth
