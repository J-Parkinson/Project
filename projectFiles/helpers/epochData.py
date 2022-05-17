import os

from projectFiles.constants import projectLoc, maxLengthSentence
from projectFiles.helpers.epochTiming import Timer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList
from projectFiles.seq2seq.plots import showPlot


class epochData:
    def __init__(self, encoder, decoder, data, embedding, curriculumLearning, hiddenLayerWidth, datasetName="",
                 batchSize=128, maxLenSentence=maxLengthSentence, noTokens=len(indicesReverseList),
                 noEpochs=1, locationToSaveTo="seq2seq/trainedModels/", learningRate=0.0003,
                 timer=Timer(), plotLosses=None, plotDevLosses=None, minLoss=999999999, minDevLoss=999999999,
                 lastIterOfDevLossImp=0, optimalEncoder=None, optimalDecoder=None, fileSaveDir=None, iGlobal=0,
                 valCheckEvery=50, clipGrad=50, decoderMultiplier=5, teacherForcingRatio=0.95):
        if plotDevLosses is None:
            plotDevLosses = []
        if plotLosses is None:
            plotLosses = []
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
        self.plotLosses = plotLosses
        self.plotDevLosses = plotDevLosses
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
        self.decoderMultiplier = decoderMultiplier
        self.teacherForcingRatio = teacherForcingRatio

    def nextEpoch(self):
        self.epochNo += 1
        return self.epochNo <= self.noEpochs

    def printPlots(self):
        print("Making plots")
        showPlot(*list(zip(*self.plotLosses)),
                 f"Training set losses for {self.datasetName} {'using' if self.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)
        showPlot(*list(zip(*self.plotDevLosses)),
                 f"Development set losses for {self.datasetName} {'using' if self.curriculumLearning.value else 'without'} curriculum learning",
                 self.fileSaveDir)

        if len(self.results) > 0:
            metrics = list(self.results[0].keys())
            metrics.remove("i")
            for metric in metrics:
                xPos = [x["i"] for x in self.results]
                yPos = [x[metric] for x in self.results]
                showPlot(xPos, yPos, metric, self.fileSaveDir)


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