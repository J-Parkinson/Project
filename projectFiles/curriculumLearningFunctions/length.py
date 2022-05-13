# Curriculum learning functions which calculate lengths of input sentence / differnece between that and output sentence

def noTokensInInput(inputSentence, _1):
    return len(inputSentence)


def differenceInLengthOfInputAndOutput(inputSentence, outputSentence):
    return len(outputSentence) - len(inputSentence)
