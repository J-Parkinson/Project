from projectFiles.evaluation.easse.calculateEASSE import calculateFleschKincaid, calculateBLEU

def fleschKincaidInput(inputSentence, _1):
    return calculateFleschKincaid([inputSentence])


def differenceInFK(inputSentence, outputSentence):
    return calculateFleschKincaid([outputSentence]) - calculateFleschKincaid([inputSentence])


def bleu(inputSentence, outputSentence):
    return calculateBLEU([inputSentence], [[outputSentence]])
