from projectFiles.evaluation.easse.calculateEASSE import calculateFleschKincaid, calculateBLEU, calculateBERTScore


def fleschKincaidInput(inputSentence, _1, _2, _3):
    return calculateFleschKincaid([inputSentence])


def differenceInFK(inputSentence, outputSentence, _1, _2):
    return calculateFleschKincaid([outputSentence]) - calculateFleschKincaid([inputSentence])


def bleu(inputSentence, outputSentence, _2, _3):
    return calculateBLEU([inputSentence], [[outputSentence]])


def bertScore(inputSentence, outputSentence, _2, _3):
    return calculateBERTScore([inputSentence], [[outputSentence]])
