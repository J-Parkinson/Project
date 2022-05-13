from projectFiles.curriculumLearningFunctions.subtlex.loadSubtlex import checkSubtlex


# Curriculum learning functions which sort by number of complex words (or sum, which takes into account both length and complexity)

def numberOfComplexWordInInput(inputSentence, _2):
    return len(list(filter(lambda x: checkSubtlex(x) > 5000, inputSentence)))


def numberOfComplexWordInInputOrOutput(inputSentence, outputSentence):
    return len(list(filter(lambda x: checkSubtlex(x) > 5000, inputSentence))) + len(
        list(filter(lambda x: checkSubtlex(x) < 5000, outputSentence)))
