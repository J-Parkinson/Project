import time

class Timer:
    def __init__(self):
        self.startTime = time.time()

    def checkTimeDiff(self):
        return time.time() - self.startTime

    def printTimeDiff(self):
        timeDiff = time.time() - self.startTime
        noSecs = int(timeDiff % 60 - timeDiff % 1)
        noMins = int((timeDiff // 60) % 60)
        noHours = int((timeDiff // 3600) % 24)
        noDays = int(timeDiff // (3600 * 24))
        if noDays:
            print(f"{noDays} days, {noHours} hours, {noMins} mins and {noSecs} secs")
        elif noHours:
            print(f"{noHours} hours, {noMins} mins and {noSecs} secs")
        elif noMins:
            print(f"{noMins} mins and {noSecs} secs")
        else:
            print(f"{noSecs} secs")

#x = Timer()
#time.sleep(4.3)
#x.printTimeDiff()
