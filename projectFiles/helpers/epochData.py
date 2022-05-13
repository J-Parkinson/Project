import csv
import os

from projectFiles.constants import projectLoc
from projectFiles.evaluation.easse.calculateEASSE import computeAll
from projectFiles.helpers.epochTiming import Timer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.constants import maxLengthSentence
from projectFiles.seq2seq.plots import showPlot


class epochData:
    def __init__(self, encoder, decoder, data, embedding, curriculumLearning, hiddenLayerWidth, datasetName="",
                 batchSize=128, maxLenSentence=maxLengthSentence, noTokens=len(indicesReverseList),
                 noEpochs=1, locationToSaveTo="seq2seq/trainedModels/", learningRate=0.01,
                 timer=Timer(), plot_losses=None, plot_dev_losses=None, minLoss=999999999, minDevLoss=999999999,
                 lastIterOfDevLossImp=0, optimalEncoder=None, optimalDecoder=None, fileSaveDir=None, iGlobal=0,
                 valCheckEvery=50, clipGrad=50):
        if plot_dev_losses is None:
            plot_dev_losses = []
        if plot_losses is None:
            plot_losses = []
        if not fileSaveDir:
            fileSaveDir = f"{projectLoc}/{locationToSaveTo}{datasetName}_CL-{curriculumLearning.name}_{embedding.name}_{timer.getStartTime().replace(':', '')}"
        os.mkdir(fileSaveDir)
        self.curriculumLearning = curriculumLearning
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.datasetName = datasetName
        self.locationToSaveTo = locationToSaveTo
        self.learningRate = learningRate
        self.timer = timer
        self.plot_losses = plot_losses
        self.plot_dev_losses = plot_dev_losses
        self.minLoss = minLoss
        self.minDevLoss = minDevLoss
        self.lastIterOfDevLossImp = lastIterOfDevLossImp
        self.optimalEncoder = optimalEncoder
        self.optimalDecoder = optimalDecoder
        self.fileSaveDir = fileSaveDir
        self.batchNoGlobal = iGlobal
        self.valCheckEvery = valCheckEvery
        self.epochNo = 0
        self.embedding = embedding
        self.results = []
        self.batchSize = batchSize
        self.maxLenSentence = maxLenSentence
        self.hiddenLayerWidth = hiddenLayerWidth
        self.noTokens = noTokens
        self.clipGrad = clipGrad
        self.noEpochs = noEpochs

    def nextEpoch(self):
        self.epochNo += 1
        return self.epochNo <= self.noEpochs

    def printPlots(self):
        print("Making plots")
        showPlot(*list(zip(*self.plot_losses)),
                 f"Training set losses for {self.datasetName} {'using' if self.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)
        showPlot(*list(zip(*self.plot_dev_losses)),
                 f"Development set losses for {self.datasetName} {'using' if self.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)

        if len(self.results) > 0:
            metrics = list(self.results[0].keys())
            metrics.remove("i")
            for metric in metrics:
                xPos = [x["i"] for x in self.results]
                yPos = [x[metric] for x in self.results]
                showPlot(xPos, yPos, metric, self.fileSaveDir)

    def savePlotData(self):
        with open(f"{self.fileSaveDir}/plotData.txt", "w+") as file:
            for x, y in self.plot_losses:
                file.write(f"{x} {y}\n")

        with open(f"{self.fileSaveDir}/plotDataDev.txt", "w+") as fileDev:
            for x, y in self.plot_dev_losses:
                fileDev.write(f"{x} {y}\n")

        if len(self.results) > 0:
            metrics = list(self.results[0].keys())
            metrics.remove("i")
            for metric in metrics:
                xPos = [x["i"] for x in self.results]
                yPos = [x[metric] for x in self.results]
                metricNameParts = metric.split("`")
                metricName = metricNameParts[0]
                if len(metricNameParts) > 1:
                    metricNameComps = metric.split("`")[1].split("/")
                    for yI, comp in enumerate(metricNameComps):
                        with open(f"{self.fileSaveDir}/{metricName}_{comp}.txt", "w+") as fileDev:
                            for x, y in zip(xPos, [yW[yI] for yW in yPos]):
                                fileDev.write(f"{x} {y}\n")
                else:
                    with open(f"{self.fileSaveDir}/{metricName}.txt", "w+") as fileDev:
                        for x, y in zip(xPos, yPos):
                            fileDev.write(f"{x} {y}\n")

    def saveTestData(self, allData):
        with open(f"{self.fileSaveDir}/evaluatedSentences.csv", 'w', encoding="utf-8") as file:
            spamWriter = csv.writer(file)
            spamWriter.writerow(
                ["Original", "Prediction"] + [f"Simplified {n}" for n in range(len(allData[0]["output"]))])
            spamWriter.writerows([[line["input"], line["predicted"]] + line["output"] for line in allData])

    def evaluateEASSE(self, allData):
        allOriginal = [v["input"] for v in allData]
        allSimplified = [v["output"] for v in allData]
        allPredicted = [v["predicted"] for v in allData]

        easseResults = computeAll(allOriginal, allSimplified, allPredicted)
        with open(f"{self.fileSaveDir}/easse.txt", "w+") as file:
            for k, v in easseResults.items():
                file.write(f"{k}:{v}\n")

    def getAttributeStr(self, iLocal):
        return f"Epoch: {self.epochNo}\n" \
               f"Sentence no / iteration in epoch: {iLocal}\n" \
               f"Global sentence no / iteration: {self.batchNoGlobal}\n" \
               f"Dataset name: {self.datasetName}\n" \
               f"Curriculum learning: {self.curriculumLearning.name}\n" \
               f"Learning rate: {self.learningRate}\n" \
               f"Time ran: {self.timer.checkTimeDiff()}\n" \
               f"Batch size: {self.batchSize}\n" \
               f"Min loss: {self.minLoss}\n" \
               f"Min dev loss: {self.minDevLoss}\n" \
               f"Check dev loss every {self.valCheckEvery} sentences\n"
