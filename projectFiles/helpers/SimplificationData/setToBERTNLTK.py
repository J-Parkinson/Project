def convertSetForEmbedding(set, embedding):
    # Import deferral done to fix circular import issues
    from projectFiles.helpers.SimplificationData.SimplificationSetBERT import simplificationSetBERT
    from projectFiles.helpers.SimplificationData.simplificationSetNLTK import simplificationSetNLTK
    from projectFiles.helpers.embeddingType import embeddingType
    if embedding == embeddingType.bert:
        set.__class__ = simplificationSetBERT
    else:
        set.__class__ = simplificationSetNLTK
    set.tokenise()
    set.addIndices()
    set.torchSet()
