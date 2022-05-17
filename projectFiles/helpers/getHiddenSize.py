from projectFiles.constants import gloveWidth
from projectFiles.helpers.embeddingType import embeddingType


def getHiddenSize(embedding, hiddenSize=512):
    if embedding == embeddingType.indices:
        return hiddenSize
    else:
        return gloveWidth
