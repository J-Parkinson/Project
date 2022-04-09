from projectFiles.evaluation.easse.calculateEASSE import calculateFleschKincaid, calculateBLEU, calculateBERTScore


def fleschKincaidInput(inputSentence, _1):
    return calculateFleschKincaid([inputSentence])


def differenceInFK(inputSentence, outputSentence):
    return calculateFleschKincaid([outputSentence]) - calculateFleschKincaid([inputSentence])


def bleu(inputSentence, outputSentence):
    return calculateBLEU([inputSentence], [[outputSentence]])


def bertScore(inputSentence, outputSentence):
    return calculateBERTScore([inputSentence], [[outputSentence]])
