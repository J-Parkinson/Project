import re

class wordFormSet:
    def __init__(self, VB="", VBD="", VBN="", VBG="", VBZ=""):
        self.VB = VB
        self.VBD = VBD
        self.VBN = VBN
        self.VBG = VBG
        self.VBZ = VBZ
        self.base = self._longestStartSharedSubsequence(VB, VBD, VBN, VBG, VBZ)

    def updateWithCheck(self, dict):
        #Assume VB is in the first pair of every new word
        if self.VB != "" and dict.get("VB",  self.VB) != self.VB:
            return False
        self.VB = dict.get("VB",  self.VB)
        self.VBD = dict.get("VBD", self.VBD)
        self.VBN = dict.get("VBN", self.VBN)
        self.VBG = dict.get("VBG", self.VBG)
        self.VBZ = dict.get("VBZ", self.VBZ)
        self.base = self._longestStartSharedSubsequence(self.VB, self.VBD, self.VBN, self.VBG, self.VBZ)
        return True

    def _longestStartSharedSubsequence(self, *vs):
        v0 = vs[0]
        for v in vs:
            v0 = self._longestStartSubsequence(v0, v)
        return v0

    def _longestStartSubsequence(self, v1, v2):
        i=0
        for i, (l1, l2) in enumerate(zip(v1, v2)):
            if l1 != l2:
                return v1[:i]
        return v1[:i]

class wordFormList:
    def __init__(self):
        self._location = "../../datasets/verb_forms/verb-form-vocab.txt"
        with open(self._location, "r", encoding="utf-8") as file:
            fileRead = file.readlines()
            fileParse = [re.match("([\w'-]*)_([\w'-]*):(\w*)_(\w*)", fileReadLine.replace("`", "").replace("^", "")).groups() for fileReadLine in fileRead]
        self._allWords = []
        currentPair = wordFormSet()
        for (w1, w2, v1, v2) in fileParse:
            dictToUse = {v1:w1, v2:w2}
            if not currentPair.updateWithCheck(dictToUse):
                self._allWords.append(currentPair)
                currentPair = wordFormSet()
                currentPair.updateWithCheck(dictToUse)
        self._allWords.append(currentPair)
        self.dictionar




wordFormList()