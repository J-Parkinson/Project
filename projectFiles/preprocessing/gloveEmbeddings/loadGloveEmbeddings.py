from random import random

from projectFiles.constants import baseLoc


def loadGloveEmbeddings():
    with open(f"{baseLoc}/datasets/glove/glove.6B.300d.txt", "r", encoding="utf-8") as glove:
        gloveLines = glove.readlines()

    gloveSplit = [gloveLine.split(" ") for gloveLine in gloveLines]
    gloveEmbeddings = {gloveLine[0]: gloveLine[1:] for gloveLine in gloveSplit}
    return gloveEmbeddings


gloveEmbeddings = loadGloveEmbeddings()
sizeOfEmbeddings = 300


def getGloveEmbeddings(word):
    try:
        # Glove only has pretrained vectors for lowered words - this will lead to many-> one relationship and will introduce errors
        # but using these embeddings should improve performance.
        return gloveEmbeddings[word.lower()]
    except:
        # Glove embedding does not exist -- we need to make one
        # Use a random embedding, and save it!
        ourEmbedding = [str(random())[:8] for _ in range(300)]
        with open(f"{baseLoc}/datasets/glove/ourEmbeddings.txt", "a") as ourEmbeddings:
            ourEmbeddings.write(f"{word.lower()} {' '.join(ourEmbedding)}\n")
        return ourEmbedding
