PAD = 0
SOS = 1
EOS = 2

indicesDict = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS}
indicesReverseList = ["<pad>", "<sos>", "<eos>"]


def getIndex(word):
    if word not in indicesDict:
        indicesReverseList.append(word)
        indicesDict[word] = len(indicesDict)
    return indicesDict[word]


def getWord(index):
    return indicesReverseList[index]
