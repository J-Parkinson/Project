import time

class Timer:
    def __init__(self):
        self.startTime = time.time()
        self.lastTimeChecked = time.time()

    def getStartTime(self):
        return time.strftime("%H:%M:%S", time.localtime())

    def checkTimeDiff(self):
        return time.time() - self.startTime

    def printTimeDiff(self):
        timeDiff = time.time() - self.startTime
        noDays, noHours, noMins, noSecs = self._calculateComps(timeDiff)
        self._printTime(noDays, noHours, noMins, noSecs)

    def _calculateComps(self, timeDiff):
        noSecs = int(timeDiff % 60 - timeDiff % 1)
        noMins = int((timeDiff // 60) % 60)
        noHours = int((timeDiff // 3600) % 24)
        noDays = int(timeDiff // (3600 * 24))
        return noDays, noHours, noMins, noSecs

    def _printTime(self, noDays, noHours, noMins, noSecs, msg="Time ran: "):
        if noDays:
            print(f"{msg} {noDays} days, {noHours} hours, {noMins} mins and {noSecs} secs")
        elif noHours:
            print(f"{msg} {noHours} hours, {noMins} mins and {noSecs} secs")
        elif noMins:
            print(f"{msg} {noMins} mins and {noSecs} secs")
        else:
            print(f"{msg} {noSecs} secs")

    def calculateTimeLeftToRun(self, currentIter, totalLength):
        timeDiff = time.time() - self.startTime
        timeLeft = timeDiff * (totalLength - currentIter) / currentIter
        noDays, noHours, noMins, noSecs = self._calculateComps(timeLeft)
        self._printTime(noDays, noHours, noMins, noSecs, "Time left:")

    def printTimeBetweenChecks(self):
        timeDiff = time.time() - self.lastTimeChecked
        self.lastTimeChecked = time.time()
        noDays, noHours, noMins, noSecs = self._calculateComps(timeDiff)
        self._printTime(noDays, noHours, noMins, noSecs, "Time iter:")


#x = Timer()
#time.sleep(4.3)
#x.printTimeDiff()
