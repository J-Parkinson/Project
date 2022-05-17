from enum import Enum

import numpy as np

from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList


class embeddingType(Enum):
    indices = 0
    glove = 1

def indicesNLTKToSentences(sentences):
    words = np.array(indicesReverseList)
    sentencesTokenized = [words[sentence] for sentence in sentences]
    sentenceNoPadding = [" ".join(sentence).split("<eos>")[0] + "<eos>" for sentence in sentencesTokenized]
    return sentenceNoPadding

# input: numpy array
def convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted):
    # This function first pairs the same inputs / predicteds with all simplifieds
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
    allPredicted = [d["predicted"] for d in returnType]
    return allInputs, allOutputs, allPredicted
