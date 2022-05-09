# This takes saved (pickled) class structure and converts to either BERT or NLTK tokenisation

def convertSetForEmbeddingAndPadding(set, embedding, maxLenSentence):
    # Import deferral done to fix circular import issues
    from projectFiles.helpers.SimplificationData.SimplificationSetBERT import simplificationSetBERT
    from projectFiles.helpers.SimplificationData.SimplificationSetNLTK import simplificationSetNLTK
    from projectFiles.helpers.embeddingType import embeddingType
    if embedding == embeddingType.bert:
        set.__class__ = simplificationSetBERT
    else:
        set.__class__ = simplificationSetNLTK
    set.addMaxSentenceLen(maxLenSentence)
    set.tokenise()
    set.addIndices()
    set.torchSet()