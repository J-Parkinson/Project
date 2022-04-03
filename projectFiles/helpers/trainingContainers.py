from projectFiles.helpers.epochTiming import Timer


class epochData:
    def __init__(self, encoder, decoder, data, datasetName="", locationToSaveTo="trainedModels/", learningRate=0.01,
                 startIter=0, timer=Timer(), plot_losses=[], plot_dev_losses=[], minLoss=999999999,
                 minDevLoss=999999999,
                 noItersSinceLastDevLossImprovement=0, optimalEncoder=None, optimalDecoder=None, fileSaveName=None,
                 iGlobal=0,
                 valCheckEvery=75):
        if not fileSaveName:
            fileSaveName = f"{locationToSaveTo}optimal_{datasetName}_{timer.getStartTime()}".replace(":", "")
        notFirstYet = startIter == 0
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.datasetName = ""
        self.locationToSaveTo = locationToSaveTo
        self.learningRate = learningRate
        self.startIter = startIter
        self.timer = timer
        self.plot_losses = plot_losses
        self.plot_dev_losses = plot_dev_losses
        self.minLoss = minLoss
        self.minDevLoss = minDevLoss
        self.noItersSinceLastDevLossImprovement = noItersSinceLastDevLossImprovement
        self.optimalEncoder = optimalEncoder
        self.optimalDecoder = optimalDecoder
        self.fileSaveName = fileSaveName
        self.notFirstYet = notFirstYet
        self.iGlobal = iGlobal
        self.valCheckEvery = valCheckEvery
