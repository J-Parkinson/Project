from random import random

import torch
from torch import optim, nn

from projectFiles.evaluation.easse.calculateEASSE import computeValidation
from projectFiles.helpers.embeddingType import embeddingType, convertDataBackToWords
from projectFiles.helpers.epochData import epochData
from projectFiles.helpers.epochTiming import Timer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import indicesReverseList, PAD
from projectFiles.seq2seq.constants import device, teacher_forcing_ratio, maxLengthSentence


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
        NLL = -torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)
        loss = NLL.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    def oneHotEncodingLossCriterion(output, expected, _):
        mask = (expected != PAD)
        output = output.squeeze()
        loss, _ = maskNLLLoss(output, expected, mask)
        return loss

    def bertEmbeddingLossCriterion(output, expected, encodedTokenized):
        # encodedTokenized is used here to determine padding masks
        mask = (encodedTokenized != PAD)
        output = output.squeeze()
        MSE = nn.MSELoss(reduction="none")
        loss = MSE(output, expected)
        loss = loss.masked_select(mask).mean()
        loss = loss.to(device)
        return loss

    if embedding == embeddingType.bert:
        return bertEmbeddingLossCriterion
    return oneHotEncodingLossCriterion


def train(batch, encoder_optimizer, decoder_optimizer, criterion, trainingMetadata):
    encoder = trainingMetadata.encoder
    decoder = trainingMetadata.decoder
    batchSize = trainingMetadata.batchSize
    maxLen = trainingMetadata.maxLenSentence
    embedding = trainingMetadata.embedding
    clip = trainingMetadata.clipGrad

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input = batch["input"]
    output = batch["output"]
    indices = batch["indicesInput"]

    # This module is self contained -- we should get batch sizes and lengths from outside

    encoder_outputs = torch.zeros(batchSize, maxLen, encoder.hidden_size, device=device)

    loss = 0

    for token in range(maxLen):
        encoder_output, encoder_hidden = encoder(
            input[:, token], encoder_hidden)
        encoder_outputs[:, token] = encoder_output[:, 0]

    decoder_input = input[:, 0]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, maxLen):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = output[:, di]  # Teacher forcing
            loss += criterion(decoder_output, output[:, di], indices[:, di])

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(1, maxLen):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoded_output = decoder_output.squeeze()
            if embedding != embeddingType.bert:
                _, decoded_output = decoded_output.topk(1)
            decoder_input = decoded_output.detach()  # detach from history as input
            loss += criterion(decoder_output, output[:, di], indices[:, di])

    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / maxLen


def validationLoss(dataLoader, criterion, trainingMetadata):
    encoder = trainingMetadata.encoder
    decoder = trainingMetadata.decoder
    batchSize = trainingMetadata.batchSize
    maxLen = trainingMetadata.maxLenSentence
    embedding = trainingMetadata.embedding
    batchNoGlobal = trainingMetadata.batchNoGlobal
    resultsGlobal = trainingMetadata.results

    loss = 0

    # allInputs = []
    allOutputs = []
    allIndices = []
    allDecoderOutputs = []

    with torch.no_grad():
        for batchNo, batch in enumerate(dataLoader):

            encoder_hidden = encoder.initHidden()

            input = batch["input"]
            # allInputs.append(input)
            output = batch["output"]
            allOutputs.append(output)
            indices = batch["indicesInput"]
            allIndices.append(indices)

            encoder_outputs = torch.zeros(batchSize, maxLen, encoder.hidden_size, device=device)

            for token in range(maxLen):
                encoder_output, encoder_hidden = encoder(
                    input[:, token], encoder_hidden)
                encoder_outputs[:, token] = encoder_output[:, 0]

            decoder_input = input[:, 0]
            decoder_outputs = []
            decoder_outputs.append(decoder_input.unsqueeze(1))

            decoder_hidden = encoder_hidden

            for di in range(1, maxLen):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoded_output = decoder_output.squeeze()
                if embedding != embeddingType.bert:
                    _, decoded_output = decoded_output.topk(1)
                decoder_outputs.append(decoder_output)
                decoder_input = decoded_output.detach()  # detach from history as input
                loss += criterion(decoder_output, output[:, di], indices[:, di])

            decoder_outputs = torch.hstack(decoder_outputs)
            allDecoderOutputs.append(decoder_outputs)

    allSimplified = torch.vstack(allOutputs).detach()
    allInputs = torch.vstack(allIndices).detach()
    allPredicted = torch.vstack(allDecoderOutputs).detach()

    allSimplified, allInputs, allPredicted = convertDataBackToWords(allSimplified, allInputs, allPredicted, embedding,
                                                                    batchSize)

    results = computeValidation(allOutputs, allIndices, allDecoderOutputs)

    results["i"] = batchNoGlobal

    resultsGlobal.append(results)

    return loss / (batchNo + 1), epochData


def trainMultipleIterations(trainingMetadata=None, encoder=None, decoder=None, allData=None, datasetName=None,
                            embedding=None, hiddenLayerWidth=None, batchSize=128, curriculumLearning=None,
                            earlyStopping=False, maxLenSentence=maxLengthSentence, noTokens=len(indicesReverseList)):
    if encoder and decoder and allData and datasetName and embedding and curriculumLearning and hiddenLayerWidth:
        trainingMetadata = epochData(encoder, decoder, allData, embedding, curriculumLearning, hiddenLayerWidth,
                                     datasetName, batchSize, earlyStopping, maxLenSentence, noTokens)

    if not epochData:
        raise Exception(
            "Inadequate input -- provide either an epochData object or encoder/decoder/trainingData/datasetName")

    while trainingMetadata.epochFinished:
        print(f"\n\nEpoch {trainingMetadata.epochNo}:")
        trainingMetadata = trainOneIteration(trainingMetadata)
        trainingMetadata.startIter = 0
        trainingMetadata.nextEpoch()

    return trainingMetadata


def trainOneIteration(trainingMetadata):
    # All metadata items in trainingMetadata (epochData) are either
    # - passed in as parameters
    # - need to be passed between epochs

    # Needs to change each epoch
    timer = Timer()

    # Local to each epoch, need not be saved
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Local to each epoch
    encoder_optimizer = optim.SGD(trainingMetadata.encoder.parameters(), lr=trainingMetadata.learningRate)
    decoder_optimizer = optim.SGD(trainingMetadata.decoder.parameters(), lr=trainingMetadata.learningRate)
    criterion = lossCriterion(trainingMetadata.embedding)

    trainingDataLoader = trainingMetadata.data.trainDL
    devDataLoader = trainingMetadata.data.devDL

    plot_loss_avg = devLoss = 0

    for batchNo, batchViews in enumerate(trainingDataLoader):
        trainingMetadata.batchNoGlobal += 1

        if trainingMetadata.checkIfEpochShouldEnd():
            trainingMetadata.epochFinished = False
            return trainingMetadata

        print(batchViews["input"].shape)

        loss = train(batchViews, encoder_optimizer, decoder_optimizer, criterion,
                     trainingMetadata)  # .encoder, trainingMetadata.decoder,
        # trainingMetadata.batchSize,
        # trainingMetadata.maxLenSentence, trainingMetadata.embedding)
        print_loss_total += loss
        plot_loss_total += loss

        #######REFACTORED UP TO HERE

        # We split print_every and plot_every here back into two
        # plot_every is controlled by batchNoGlobal, print_every is controlled by batchNo

        if trainingMetadata.batchNoGlobal % trainingMetadata.valCheckEvery == 0 and trainingMetadata.batchNoGlobal:
            print_loss_avg = print_loss_total / trainingMetadata.valCheckEvery
            trainingMetadata.minLoss = min(trainingMetadata.minLoss, print_loss_total)

            # Calculate validation loss
            devLoss, trainingMetadata = validationLoss(devDataLoader, criterion, trainingMetadata)
            trainingMetadata.minDevLoss = min(trainingMetadata.minDevLoss, devLoss)

            if trainingMetadata.minDevLoss == devLoss:
                trainingMetadata.lastIterOfDevLossImp = trainingMetadata.batchNoGlobal
                trainingMetadata.optimalEncoder = trainingMetadata.encoder.state_dict()
                trainingMetadata.optimalDecoder = trainingMetadata.decoder.state_dict()
                with open(f"{trainingMetadata.fileSaveDir}/epochRun.txt", "w+") as file:
                    file.write(trainingMetadata.getAttributeStr(batchNo))
                torch.save(trainingMetadata.optimalEncoder, f"{trainingMetadata.fileSaveDir}/encoder.pt")
                torch.save(trainingMetadata.optimalDecoder, f"{trainingMetadata.fileSaveDir}/decoder.pt")

            print_loss_total = 0

            print("_____________________")
            print(f"Batch {batchNo + 1}")
            timer.printTimeDiff()
            timer.printTimeBetweenChecks()
            print(f"Loss average: {print_loss_avg}")
            print(
                f"Min avg loss per {trainingMetadata.valCheckEvery} iterations: {trainingMetadata.minLoss / trainingMetadata.valCheckEvery}")
            print(f"Validation loss: {devLoss}")

            plot_loss_avg = plot_loss_total / trainingMetadata.valCheckEvery
            trainingMetadata.plot_losses.append((trainingMetadata.batchNoGlobal + 1, plot_loss_avg))
            plot_loss_total = 0

            trainingMetadata.plot_dev_losses.append((trainingMetadata.batchNoGlobal + 1, devLoss))

    trainingMetadata.plot_losses.append((trainingMetadata.batchNoGlobal + 1, plot_loss_avg))
    trainingMetadata.plot_dev_losses.append((trainingMetadata.batchNoGlobal + 1, devLoss))

    return trainingMetadata
