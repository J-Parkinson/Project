PAD = 0
SOS = 1
EOS = 2

indicesDict = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS}
indicesReverseList = ["<pad>", "<sos>", "<eos>"]
wordToCount = {"<eos>": 0}


def reinitialiseEmbeddings():
    indicesDict.clear()
    indicesDict.update({"<pad>": PAD, "<sos>": SOS, "<eos>": EOS})
    indicesReverseList.clear()
    indicesReverseList.extend(["<pad>", "<sos>", "<eos>"])
    wordToCount.clear()
    wordToCount.update({"<eos>": 0})

def getIndex(word):
    if word not in indicesDict:
        indicesReverseList.append(word)
        indicesDict[word] = len(indicesDict)
        wordToCount[word] = 1
    else:
        wordToCount[word] += 1
    return indicesDict[word]


def getWord(index):
    return indicesReverseList[index]


def getCount(word):
    return wordToCount[word]
