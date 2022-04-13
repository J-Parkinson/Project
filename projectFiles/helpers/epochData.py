import os

from projectFiles.checkers.getTrainDevTestSizes import sizes
from projectFiles.constants import projectLoc
from projectFiles.evaluation.easse.calculateEASSE import computeAll
from projectFiles.helpers.DatasetToLoad import name2DTL
from projectFiles.helpers.epochTiming import Timer
from projectFiles.seq2seq.plots import showPlot


class epochData:
    def __init__(self, encoder, decoder, data, embedding, datasetName="", locationToSaveTo="seq2seq/trainedModels/",
                 learningRate=0.01, startIter=0, timer=Timer(), plot_losses=None, plot_dev_losses=None,
                 minLoss=999999999, minDevLoss=999999999, lastIterOfDevLossImp=0, optimalEncoder=None,
                 optimalDecoder=None, fileSaveDir=None, iGlobal=0, valCheckEvery=50, earlyStopAfterNoImpAfterIter=None):
        if plot_dev_losses is None:
            plot_dev_losses = []
        if plot_losses is None:
            plot_losses = []
        if not fileSaveDir:
            fileSaveDir = f"{projectLoc}/{locationToSaveTo}{datasetName}_CL-{data.train.curriculumLearning.name}_{embedding.name}_{timer.getStartTime().replace(':', '')}"
        if not earlyStopAfterNoImpAfterIter:
            earlyStopAfterNoImpAfterIter = sizes[name2DTL(datasetName)][0] // 4
        os.mkdir(fileSaveDir)
        notFirstYet = startIter == 0
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.datasetName = datasetName
        self.locationToSaveTo = locationToSaveTo
        self.learningRate = learningRate
        self.startIter = startIter
        self.timer = timer
        self.plot_losses = plot_losses
        self.plot_dev_losses = plot_dev_losses
        self.minLoss = minLoss
        self.minDevLoss = minDevLoss
        self.lastIterOfDevLossImp = lastIterOfDevLossImp
        self.optimalEncoder = optimalEncoder
        self.optimalDecoder = optimalDecoder
        self.fileSaveDir = fileSaveDir
        self.notFirstYet = notFirstYet
        self.iGlobal = iGlobal
        self.valCheckEvery = valCheckEvery
        self.epochFinished = True
        self.earlyStopAfterNoImpAfterIter = earlyStopAfterNoImpAfterIter
        self.epochNo = 1
        self.embedding = embedding

    def nextEpoch(self):
        self.epochNo += 1

    def checkIfEpochShouldEnd(self):
        return (self.iGlobal - self.lastIterOfDevLossImp) > self.earlyStopAfterNoImpAfterIter

    def printPlots(self):
        print("Making plots")
        showPlot(*list(zip(*self.plot_losses)),
                 f"Training set losses for {self.datasetName} {'using' if self.data.train.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)
        showPlot(*list(zip(*self.plot_dev_losses)),
                 f"Development set losses for {self.datasetName} {'using' if self.data.train.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)

    def savePlotData(self):
        with open(f"{self.fileSaveDir}/plotData.txt", "w+") as file:
            for x, y in self.plot_losses:
                file.write(f"{x} {y}\n")

        with open(f"{self.fileSaveDir}/plotDataDev.txt", "w+") as fileDev:
            for x, y in self.plot_dev_losses:
                fileDev.write(f"{x} {y}\n")

    def saveTestData(self):
        with open(f"{self.fileSaveDir}/originalSentences.txt", "w+") as file:
            for setV in self.data.test:
                file.write(f"{setV.original}\n")

        with open(f"{self.fileSaveDir}/predictedSimplifications.txt", "w+") as file:
            for setV in self.data.test:
                file.write(f"{setV.predicted}\n")

    def evaluateEASSE(self):
        easseResults = computeAll(self.data.test)
        with open(f"{self.fileSaveDir}/easse.txt", "w+") as file:
            for k, v in easseResults.items():
                file.write(f"{k}:{v}\n")

    def getAttributeStr(self, iLocal):
        return f"Epoch: {self.epochNo}\n" \
               f"Sentence no / iteration in epoch: {iLocal}\n" \
               f"Global sentence no / iteration: {self.iGlobal}\n" \
               f"Dataset name: {self.datasetName}\n" \
               f"Curriculum learning: {self.data.train.curriculumLearning.name}\n" \
               f"Learning rate: {self.learningRate}\n" \
               f"Time ran: {self.timer.checkTimeDiff()}\n" \
               f"Min loss: {self.minLoss}\n" \
               f"Min dev loss: {self.minDevLoss}\n" \
               f"Check dev loss every {self.valCheckEvery} sentences\n" \
               f"Early stop after {self.earlyStopAfterNoImpAfterIter} sentences of no validation loss improvement"
