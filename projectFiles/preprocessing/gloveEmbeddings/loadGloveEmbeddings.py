from projectFiles.constants import baseLoc


def loadGloveEmbeddings():
    with open(f"{baseLoc}/datasets/glove/glove.6B.300d.txt", "r") as glove:
        gloveLines = glove.readlines()

    gloveSplit = [gloveLine.split(" ") for gloveLine in gloveLines]
    gloveEmbeddings = {gloveLine[0]: gloveLine[1:] for gloveLine in gloveSplit}
    return gloveEmbeddings
