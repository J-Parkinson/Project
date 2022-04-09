from projectFiles.curriculumLearningFunctions.subtlex.loadSubtlex import checkSubtlex


def numberOfComplexWordInInput(inputSentence, _2):
    return len(list(filter(lambda x: checkSubtlex(x) > 5000, inputSentence)))


def numberOfComplexWordInInputOrOutput(inputSentence, outputSentence):
    return len(list(filter(lambda x: checkSubtlex(x) > 5000, inputSentence))) + len(
        list(filter(lambda x: checkSubtlex(x) < 5000, outputSentence)))


def sumOfIndices(inputSentence, outputSentence):
    return sum([checkSubtlex(x) for x in inputSentence]) + sum([checkSubtlex(x) for x in outputSentence])
