from projectFiles.constants import baseLoc


def loadSubtlexData():
    subtlexLoc = f"{baseLoc}/datasets/subtlex/SUBTLEX_frequency.csv"
    with open(subtlexLoc, "r") as subtlex:
        subtlexRead = subtlex.readlines()
        subtlexData = [line.split(",", 3)[:3] for line in subtlexRead][1:]
        subtlexData = [(x[0], int(x[1]), int(x[2])) for x in subtlexData]
    subtlexData.sort(key=lambda x: x[1], reverse=True)
    subtlexData.sort(key=lambda x: x[2], reverse=True)
    subtlexData = {x[0]: y for y, x in enumerate(subtlexData)}
    return subtlexData


_subtlexData = loadSubtlexData()


def checkSubtlex(word):
    try:
        return _subtlexData[word]
    except:
        return len(_subtlexData)
