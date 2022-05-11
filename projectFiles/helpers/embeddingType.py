from enum import Enum

import numpy as np

from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model, tokenizer
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
    sentencesTokenized = words[sentences]
    sentenceNoPadding = [" ".join(filter(lambda x: x != 0, sentence)) for sentence in sentencesTokenized]
    return sentenceNoPadding


def bertToSentences(sentences, batchSize, manualPad=False):
    noIters = sentences.shape[0] // batchSize
    allTokenized = []
    for x in range(noIters):
        bertResult = model.get_output_embeddings()(sentences[x * batchSize:(x + 1) * batchSize])
        allTokenized += [decodedSentence.softmax(0).argmax(1).cpu().numpy() for decodedSentence in bertResult]
    return indicesBertToSentences(np.array(allTokenized), manualPad)


# input: numpy array
def convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted, embedding, batchSize):
    # This function first pairs the same inputs / predicteds with all simplifieds

    if embedding == embeddingType.bert:
        allInputIndicesSentenced = indicesBertToSentences(allInputIndices.squeeze())
        allOutputIndicesSentenced = indicesBertToSentences(allOutputIndices.squeeze())
        allPredictedSentenced = bertToSentences(allPredicted.squeeze(), batchSize, manualPad=True)
    else:
        allInputIndicesSentenced = indicesNLTKToSentences(allInputIndices.squeeze())
        allOutputIndicesSentenced = indicesNLTKToSentences(allOutputIndices.squeeze())
        allPredictedSentenced = indicesNLTKToSentences(allPredicted.squeeze())

    allPredictedSentenced = allPredictedSentenced[:len(allInputIndicesSentenced)]

    inputSentenceNo = [0]
    for a in range(len(allInputIndicesSentenced) - 1):
        inputSentenceNo += [inputSentenceNo[-1] + int(allInputIndicesSentenced[a] != allInputIndicesSentenced[a + 1])]

    returnType = [{"input": "", "output": [], "predicted": []} for _ in range(inputSentenceNo[-1] + 1)]
    for inp, out, pred, sentNo in zip(allInputIndicesSentenced, allOutputIndicesSentenced, allPredictedSentenced,
                                      inputSentenceNo):
        returnType[sentNo]["input"] = inp
        returnType[sentNo]["output"].append(out)
        returnType[sentNo]["predicted"].append(pred)

    allInputs = [d["input"] for d in returnType]
    allOutputs = [d["output"] for d in returnType]
    allPredicted = [d["predicted"][0] for d in returnType]  # TODO: CHECK BUG HERE
    return allInputs, allOutputs, allPredicted
