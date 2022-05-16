# This takes saved (pickled) class structure and converts to either NLTK tokenisation
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import getCount


def convertSetForEmbeddingAndPadding(set, embedding):
    # Import deferral done to fix circular import issues
    from projectFiles.helpers.SimplificationData.SimplificationSetNLTK import simplificationSetNLTK
    set.__class__ = simplificationSetNLTK
    set.tokenise()
    set.addIndices()
    return set


def convertSetForEmbeddingAndPaddingAndFlagLong(set, embedding, maxLenSentence, minOccurences):
    # Import deferral done to fix circular import issues
    set = convertSetForEmbeddingAndPadding(set, embedding)

    # Remove sets with ANY too long sentences
    maxLenSentenceInSet = max([len(set.originalTokenized)] + [len(simple) for simple in set.allSimpleTokenized])
    if maxLenSentenceInSet > maxLenSentence:
        return None

    # Remove sentences with uncommon tokens in them
    # Delete sets where original sentence has weird tokens in it
    if sum([getCount(word) < minOccurences for word in set.originalTokenized]) > 0:
        return None

    # Delete sets where all simple sentences have weird tokens in them
    if all([sum([getCount(word) < minOccurences for word in simple]) > 0 for simple in set.allSimpleTokenized]):
        return None

    # Delete only sentences that have weird tokens in them
    set.allSimpleTokenized = [simple for simple in set.allSimpleTokenized if
                              sum([getCount(word) < minOccurences for word in simple]) == 0]

    return set
