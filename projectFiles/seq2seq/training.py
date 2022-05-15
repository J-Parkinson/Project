from random import random

import torch
from torch import optim, nn

from projectFiles.evaluation.easse.calculateEASSE import computeValidation
from projectFiles.helpers.DatasetSplits import datasetSplits
from projectFiles.helpers.embeddingType import embeddingType, convertDataBackToWords
from projectFiles.helpers.epochData import epochData
from projectFiles.helpers.epochTiming import Timer
from projectFiles.helpers.getSpecialTokens import getDecoderInput
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList, PAD
from projectFiles.seq2seq.constants import device, maxLengthSentence


# Batching now requires us to handle losses in a more interesting way (to deal with padding)
def lossCriterion(embedding):
    # We can either calculate losses ignoring tokens after EOS or not
    # Since we ignore tokens after EOS (and padding is removed anyway) we ignore losses after EOS
    # Hence we only compute mean over those tokens inc EOS
    # Which EOS to pick? We pick the PREDICTED EOS as the end point
    # a) it's always there
    # b) no worries if EOS appears twice in decoder output

    # https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
    def maskNLLLoss(inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    def bertEmbeddingLossCriterion(output, expected, encodedTokenized):
        # encodedTokenized is used here to determine padding masks
        mask = (encodedTokenized != PAD)
        output = output.squeeze()
        MSE = nn.MSELoss(reduction="none")
        loss = MSE(output, expected)
        loss = loss.masked_select(mask).mean()
        loss = torch.nan_to_num(loss)
        loss = loss.to(device)
        return loss, 1

    if embedding == embeddingType.bert:
        return bertEmbeddingLossCriterion
    return maskNLLLoss


def train(batch, encoderOptimizer, decoderOptimizer, criterion, trainingMetadata):
    encoder = trainingMetadata.encoder
    encoder.train()
    decoder = trainingMetadata.decoder
    decoder.train()
    batchSize = trainingMetadata.batchSize
    embedding = trainingMetadata.embedding
    clip = trainingMetadata.clipGrad
    teacherForcingRatio = trainingMetadata.teacherForcingRatio

    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    inputEmbeddings = batch["inputEmbeddings"]
    outputEmbeddings = batch["outputEmbeddings"]
    inputIndices = batch["inputIndices"]
    outputIndices = batch["outputIndices"]
    lengths = batch["lengths"]
    maxSimplifiedLength = batch["maxSimplifiedLength"]
    mask = batch["mask"]

    loss = 0
    weightedLoss = 0
    noTokensInBatch = 0

    encoderOutputs, encoderHidden = encoder(inputEmbeddings, lengths)

    decoderInput = getDecoderInput(embedding, batchSize)
    decoderHidden = encoderHidden[:decoder.noLayers]

    useTeacherForcing = True if random() < teacherForcingRatio else False

    if useTeacherForcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(maxSimplifiedLength):
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            decoderInput = outputEmbeddings[di].view(1, -1)
            addLoss, noTokens = criterion(decoderOutput, outputEmbeddings[di], mask[di])
            loss += addLoss
            weightedLoss += addLoss.item() * noTokens
            noTokensInBatch += noTokens

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(maxSimplifiedLength):
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)

            if embedding != embeddingType.bert:
                _, topi = decoderOutput.topk(1)
                decoderInput = torch.tensor([[topi[i][0] for i in range(batchSize)]], device=device)

            addLoss, noTokens = criterion(decoderOutput, outputEmbeddings[di], mask[di])
            loss += addLoss
            weightedLoss += addLoss.item() * noTokens
            noTokensInBatch += noTokens

    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoderOptimizer.step()
    decoderOptimizer.step()

    print(f"Training loss: {weightedLoss}")

    if embedding == embeddingType.bert:
        return weightedLoss
    else:
        return weightedLoss / noTokensInBatch


def validationEvaluationLoss(trainingMetadata, mode, criterion=None, dataLoader=None):
    dataLoader = dataLoader or trainingMetadata.data.testDL
    print(f"{'Validation' if mode == datasetSplits.dev else 'Evaluation'} calculating----------------------------")
    encoder = trainingMetadata.encoder
    encoder.eval()
    decoder = trainingMetadata.decoder
    decoder.eval()
    batchSize = trainingMetadata.batchSize
    embedding = trainingMetadata.embedding
    batchNoGlobal = trainingMetadata.batchNoGlobal
    resultsGlobal = trainingMetadata.results

    loss = 0
    allWeightedLosses = 0
    noTokensInDevSet = 0

    # allInputs = []
    # allOutputs = []
    allInputIndices = []
    allOutputIndices = []
    allDecoderOutputs = []

    with torch.no_grad():
        for batchNo, batch in enumerate(dataLoader):
            inputEmbeddings = batch["inputEmbeddings"]
            outputEmbeddings = batch["outputEmbeddings"]
            inputIndices = batch["inputIndices"]
            allInputIndices.append(inputIndices)
            outputIndices = batch["outputIndices"]
            allOutputIndices.append(outputIndices)
            lengths = batch["lengths"]
            maxSimplifiedLength = batch["maxSimplifiedLength"]
            mask = batch["mask"]

            encoderOutputs, encoderHidden = encoder(inputEmbeddings, lengths)

            decoderOutputs = []
            decoderInput = getDecoderInput(embedding, batchSize)
            decoderHidden = encoderHidden[:decoder.noLayers]

            noTokensInBatch = 0
            weightedLoss = 0

            for di in range(maxSimplifiedLength):
                decoderOutput, decoderHidden, decoderAttention = decoder(
                    decoderInput, decoderHidden, encoderOutputs)

                if embedding != embeddingType.bert:
                    _, topi = decoderOutput.topk(1)
                    decoderInput = torch.tensor([[topi[i][0] for i in range(batchSize)]], device=device)

                decoderOutputs.append(decoderInput)

                if mode == datasetSplits.dev:
                    addLoss, noTokens = criterion(decoderOutput, outputEmbeddings[di], mask[di])
                    loss += addLoss
                    weightedLoss += addLoss.item() * noTokens
                    noTokensInBatch += noTokens

            decoderOutputs = torch.vstack(decoderOutputs).swapaxes(0, 1)
            allDecoderOutputs.append(decoderOutputs)
            if mode == datasetSplits.dev:
                allWeightedLosses += weightedLoss
                noTokensInDevSet += noTokensInBatch

    allPredicted = [sentence.cpu().detach().numpy() for batch in allDecoderOutputs for sentence in batch]
    allInputIndices = [batch.swapaxes(0, 1) for batch in allInputIndices]
    allInputIndices = [sentence.cpu().detach().numpy() for batch in allInputIndices for sentence in batch]
    allOutputIndices = [batch.swapaxes(0, 1) for batch in allOutputIndices]
    allOutputIndices = [sentence.cpu().detach().numpy() for batch in allOutputIndices for sentence in batch]

    allInputs, allOutputs, allPredicted = convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted,
                                                                 embedding)

    if mode == datasetSplits.dev:
        results = computeValidation(allInputs, allOutputs, allPredicted)

        results["i"] = batchNoGlobal

        resultsGlobal.append(results)

        print("Validation calculated-----------------------------")

        if embedding == embeddingType.bert:
            return allWeightedLosses, trainingMetadata
        else:
            return allWeightedLosses / noTokensInDevSet, trainingMetadata
    else:
        allData = [{"input": inp, "output": out, "predicted": pred} for inp, out, pred in
                   zip(allInputs, allOutputs, allPredicted)]

        return allData

def trainOneIteration(trainingMetadata):
    # All metadata items in trainingMetadata (epochData) are either
    # - passed in as parameters
    # - need to be passed between epochs

    # Needs to change each epoch
    timer = Timer()

    # Local to each epoch, need not be saved
    printLossTotal = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Local to each epoch
    encoderOptimizer = optim.Adam(trainingMetadata.encoder.parameters(), lr=trainingMetadata.learningRate)
    decoderOptimizer = optim.Adam(trainingMetadata.decoder.parameters(), lr=trainingMetadata.learningRate
                                                                            * trainingMetadata.decoderMultiplier)
    criterion = lossCriterion(trainingMetadata.embedding)

    trainingDataLoader = trainingMetadata.data.trainDL
    devDataLoader = trainingMetadata.data.devDL

    plot_loss_avg = devLoss = 0

    for batchNo, batchViews in enumerate(trainingDataLoader):
        trainingMetadata.batchNoGlobal += 1

        print(f"Batch {batchNo + 1} running...")

        loss = train(batchViews, encoderOptimizer, decoderOptimizer, criterion,
                     trainingMetadata)  # .encoder, trainingMetadata.decoder,
        # trainingMetadata.batchSize,
        # trainingMetadata.maxLenSentence, trainingMetadata.embedding)
        printLossTotal += loss
        plot_loss_total += loss

        #######REFACTORED UP TO HERE

        # We split print_every and plot_every here back into two
        # plot_every is controlled by batchNoGlobal, print_every is controlled by batchNo

        if trainingMetadata.batchNoGlobal % trainingMetadata.valCheckEvery == 0:
            printLossAvg = printLossTotal / trainingMetadata.valCheckEvery
            trainingMetadata.minLoss = min(trainingMetadata.minLoss, printLossTotal)

            # Calculate validation loss
            devLoss, trainingMetadata = validationEvaluationLoss(trainingMetadata, datasetSplits.dev, criterion,
                                                                 devDataLoader)
            trainingMetadata.minDevLoss = min(trainingMetadata.minDevLoss, devLoss)

            if trainingMetadata.minDevLoss == devLoss:
                trainingMetadata.lastIterOfDevLossImp = trainingMetadata.batchNoGlobal
                trainingMetadata.optimalEncoder = trainingMetadata.encoder.state_dict()
                trainingMetadata.optimalDecoder = trainingMetadata.decoder.state_dict()
                with open(f"{trainingMetadata.fileSaveDir}/epochRun.txt", "w+") as file:
                    file.write(trainingMetadata.getAttributeStr(batchNo))
                torch.save(trainingMetadata.optimalEncoder, f"{trainingMetadata.fileSaveDir}/encoder.pt")
                torch.save(trainingMetadata.optimalDecoder, f"{trainingMetadata.fileSaveDir}/decoder.pt")

            printLossTotal = 0

            print("_____________________")
            print(f"Batch {batchNo + 1}")
            timer.printTimeDiff()
            timer.printTimeBetweenChecks()
            print(f"Loss average: {printLossAvg}")
            print(
                f"Min avg loss per {trainingMetadata.valCheckEvery} iterations: {trainingMetadata.minLoss / trainingMetadata.valCheckEvery}")
            print(f"Validation loss: {devLoss}")

            plot_loss_avg = plot_loss_total / trainingMetadata.valCheckEvery
            trainingMetadata.plotLosses.append((trainingMetadata.batchNoGlobal + 1, plot_loss_avg))
            plot_loss_total = 0

            trainingMetadata.plotDevLosses.append((trainingMetadata.batchNoGlobal + 1, devLoss))

    trainingMetadata.plotLosses.append((trainingMetadata.batchNoGlobal + 1, plot_loss_avg))
    trainingMetadata.plotDevLosses.append((trainingMetadata.batchNoGlobal + 1, devLoss))

    return trainingMetadata


def trainMultipleIterations(trainingMetadata=None, encoder=None, decoder=None, allData=None, datasetName=None,
                            embedding=None, hiddenLayerWidth=None, batchSize=128, curriculumLearning=None,
                            maxLenSentence=maxLengthSentence, noTokens=len(indicesReverseList),
                            noEpochs=1, batchesBetweenValidation=50):
    if encoder and decoder and allData and datasetName and embedding and curriculumLearning and hiddenLayerWidth:
        trainingMetadata = epochData(encoder, decoder, allData, embedding, curriculumLearning, hiddenLayerWidth,
                                     datasetName, batchSize, maxLenSentence, noTokens, noEpochs,
                                     valCheckEvery=batchesBetweenValidation)

    if not epochData:
        raise Exception(
            "Inadequate input -- provide either an epochData object or encoder/decoder/trainingData/datasetName")

    while trainingMetadata.nextEpoch():
        print(f"\n\nEpoch {trainingMetadata.epochNo}:")
        trainingMetadata = trainOneIteration(trainingMetadata)

    return trainingMetadata
