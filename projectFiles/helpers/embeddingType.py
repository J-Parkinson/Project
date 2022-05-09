from enum import Enum

import numpy as np
import torch

from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model, tokenizer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import getWord


class embeddingType(Enum):
    indices = 0
    glove = 1
    bert = 2


def indicesBertToSentences(sentences):
    # Remove padding
    sentences = tokenizer.batch_decode(sentences.squeeze())
    sentencesNoPadding = np.array(
        [" ".join(filter(lambda x: x != "[PAD]", sentence.split(" "))) for sentence in sentences])
    return sentencesNoPadding


def indicesNLTKToSentences(sentences):
    sentenceNoPadding = [z for p, z in enumerate(sentences) if sentences[p:] != [0 for _ in range(len(sentences[p:]))]]
    sentenceTokenized = np.array([getWord(ind) for ind in sentenceNoPadding])
    return sentenceTokenized


def bertToSentences(sentences, batchSize):
    noIters = sentences.shape[0] // batchSize
    allTokenized = []
    for x in range(noIters):
        bertResult = model.get_output_embeddings()(sentences[x * batchSize:(x + 1) * batchSize - 1])
        allTokenized += [decodedSentence.softmax(0).argmax(1).cpu().numpy() for decodedSentence in bertResult]
    return indicesBertToSentences(np.array(allTokenized))


# input: numpy array
def convertDataBackToWords(allSimplified, allInputs, allPredicted, embedding, batchSize):
    # This function first pairs the same inputs / predicteds with all simplifieds

    allData = torch.hstack((np.expand_dims(allInputs, axis=1),
                            np.expand_dims(allPredicted, axis=1),
                            np.expand_dims(allSimplified, axis=1)))
    dataKeys = [str(sentence) for sentence in allData[:, 0]]
    dataDict = {k: {"original": "", "predicted": "", "simplified": []} for k in set(dataKeys)}
    for setOfData in allData:
        dataDict[str(setOfData[0])]["original"] = setOfData[0]
        dataDict[set(setOfData[0])]["predicted"] = setOfData[1]
        dataDict[str(setOfData[0])]["simplified"].append(setOfData[2])

    dataDict = list(dataDict.values())

    if embedding == embeddingType.indices:
        pass
    elif embedding == embeddingType.glove:
        pass
    else:
        pass

    # Remove padding
    return None
