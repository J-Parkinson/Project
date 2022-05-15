from enum import Enum

import numpy as np

from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import tokenizer, embeddingOutput
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList


class embeddingType(Enum):
    indices = 0
    glove = 1
    bert = 2


def indicesBertToSentences(sentences, manualPad=False):
    # Remove padding
    sentences = tokenizer.batch_decode(sentences.squeeze())
    if not manualPad:
        sentences = list(filter(lambda x: x != "",
                                [" ".join(filter(lambda x: x != "[PAD]", sentence.split(" "))) for sentence in
                                 sentences]))
        # What if [SEP] is followed by garbage?
    sentences = [sentence.split("[SEP]")[0] + "[SEP]" for sentence in sentences]
    return sentences


def indicesNLTKToSentences(sentences):
    words = np.array(indicesReverseList)
    sentencesTokenized = [words[sentence] for sentence in sentences]
    sentenceNoPadding = [" ".join(sentence).split("<eos>")[0] + "<eos>" for sentence in sentencesTokenized]
    return sentenceNoPadding


def bertToSentences(sentences, batchSize, manualPad=False):
    noIters = sentences.shape[0] // batchSize
    allTokenized = []
    for x in range(noIters):
        bertResult = embeddingOutput(sentences[x * batchSize:(x + 1) * batchSize])
        allTokenized += [decodedSentence.softmax(0).argmax(1).cpu().numpy() for decodedSentence in bertResult]
    return indicesBertToSentences(np.array(allTokenized), manualPad)


# input: numpy array
def convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted, embedding):
    # This function first pairs the same inputs / predicteds with all simplifieds

    if embedding == embeddingType.bert:
        allInputIndicesSentenced = indicesBertToSentences(allInputIndices)
        allOutputIndicesSentenced = indicesBertToSentences(allOutputIndices)
        allPredictedSentenced = bertToSentences(allPredicted, 64, manualPad=True)
    else:
        allInputIndicesSentenced = indicesNLTKToSentences(allInputIndices)
        allOutputIndicesSentenced = indicesNLTKToSentences(allOutputIndices)
        allPredictedSentenced = indicesNLTKToSentences(allPredicted)

    returnType = []
    for inp, out, pred in zip(allInputIndicesSentenced, allOutputIndicesSentenced, allPredictedSentenced):
        found = False
        for i in range(len(returnType)):
            if returnType[i]["input"] == inp:
                returnType[i]["output"].append(out)
                found = True
                break
        if not found:
            returnType.append({"input": inp, "output": [out], "predicted": pred})

    allInputs = [d["input"] for d in returnType]
    allOutputs = [d["output"] for d in returnType]
    allPredicted = [d["predicted"][0] for d in returnType]  # TODO: CHECK BUG HERE
    return allInputs, allOutputs, allPredicted
