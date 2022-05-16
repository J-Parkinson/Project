from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.seq2seq.constants import gloveWidth

def getHiddenSize(hiddenSize, embedding):
    if embedding == embeddingType.indices:
        return hiddenSize
    else:
        return gloveWidth
