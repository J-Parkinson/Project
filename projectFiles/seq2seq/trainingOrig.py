from random import random

import torch
from torch import optim, nn

from projectFiles.helpers.epochData import teacherForcingRatio
from projectFiles.helpers.epochTiming import Timer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS, EOS
from projectFiles.seq2seq.constants import device, maxLengthSentence
from projectFiles.seq2seq.plots import showPlot


def train(inputTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion,
          maxLength=maxLengthSentence):
    encoderHidden = encoder.initHidden()

    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    input_length = inputTensor.size(0)
    target_length = targetTensor.size(0)

    encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize, device=device)

    loss = 0

    for ei in range(input_length):
        encoderOutput, encoderHidden = encoder(
            inputTensor[ei], encoderHidden)
        encoderOutputs[ei] = encoderOutput[0, 0]

    decoderInput = torch.tensor([[SOS]], device=device)

    decoderHidden = encoderHidden

    useTeacherForcing = True if random() < teacherForcingRatio else False

    if useTeacherForcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            loss += criterion(decoderOutput, targetTensor[di])
            decoderInput = targetTensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoderOutput, targetTensor[di])
            if decoderInput.item() == EOS:
                break

    loss.backward()

    encoderOptimizer.step()
    decoderOptimizer.step()

    return loss.item() / target_length


def validationLoss(inputTensors, targetTensors_set, encoder, decoder, criterion, maxLength=maxLengthSentence):
    losses = []
    with torch.no_grad():
        for inputTensor, targetTensors in zip(inputTensors, targetTensors_set):
            for targetTensor in targetTensors:
                encoderHidden = encoder.initHidden()

                input_length = inputTensor.size(0)
                target_length = targetTensor.size(0)

                encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize, device=device)

                loss = 0

                for ei in range(input_length):
                    encoderOutput, encoderHidden = encoder(
                        inputTensor[ei], encoderHidden)
                    encoderOutputs[ei] = encoderOutput[0, 0]

                decoderInput = torch.tensor([[SOS]], device=device)

                decoderHidden = encoderHidden

                # We do not use teacher forcing during evaluation
                for di in range(target_length):
                    decoderOutput, decoderHidden, decoderAttention = decoder(
                        decoderInput, decoderHidden, encoderOutputs)
                    topv, topi = decoderOutput.topk(1)
                    decoderInput = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoderOutput, targetTensor[di])
                    if decoderInput.item() == EOS:
                        break
                losses.append(loss.item() / target_length)
    return sum(losses) / len(losses)


def trainIters(encoder, decoder, allData, datasetName, locationToSaveTo="trainedModels/", print_every=150,
               plot_every=100, learning_rate=0.01, startIter=0):
    timer = Timer()
    plotLosses = []
    plotDevLosses = []
    printLossTotal = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoderOptimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoderOptimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    minLoss = 999999999999999999999999999999999999
    minDevLoss = 999999999999999999999999999999999999
    optimalEncoder = None
    optimalDecoder = None
    fileSaveName = f"{locationToSaveTo}optimal_{datasetName}_{timer.getStartTime()}".replace(":", "")
    notFirstYet = startIter == 0
    i = 0

    trainingData = allData.train
    validationData = allData.dev
    curriculumLearning = True

    for m, pair in enumerate(trainingData):
        inputTensor = pair.originalTorch
        targetTensors = pair.allSimpleTorches
        j = 0
        l = 0
        while i < startIter and j < len(targetTensors):
            i += 1
            j += 1
        if j == len(targetTensors):
            continue

        for k, targetTensor in enumerate(targetTensors):
            i += 1
            if i > 300:
                continue
            print(i)

            if k < j:
                continue

            l += len(targetTensors)

            loss = train(inputTensor, targetTensor, encoder,
                         decoder, encoderOptimizer, decoderOptimizer, criterion)
            printLossTotal += loss
            plot_loss_total += loss

            if (i + 1) % print_every == 0:
                if not notFirstYet:
                    notFirstYet = True
                else:
                    printLossAvg = printLossTotal / print_every
                    minLoss = min(minLoss, printLossTotal)

                    # Calculate validation loss
                    devLoss = validationLoss([dataVal.originalTorch for dataVal in validationData],
                                             [dataVal.allSimpleTorches for dataVal in validationData],
                                             encoder, decoder, criterion)
                    minDevLoss = min(minDevLoss, devLoss)

                    if minDevLoss == devLoss:
                        optimalEncoder = encoder.state_dict()
                        optimalDecoder = decoder.state_dict()
                        with open(f"{fileSaveName}.txt", "w+") as file:
                            file.write(f"Iteration {i + 1}\nDataset: {datasetName}")
                        torch.save(optimalEncoder, f"{fileSaveName}_encoder.pt")
                        torch.save(optimalDecoder, f"{fileSaveName}_decoder.pt")

                    printLossTotal = 0

                    print("_____________________")
                    print(f"Iteration {i + 1}")
                    timer.printTimeDiff()
                    timer.printTimeBetweenChecks()
                    timer.calculateTimeLeftToRun(i + 1, len(trainingData) * l / m)
                    print(f"Loss average: {printLossAvg}")
                    print(f"Min avg loss per {print_every} iterations: {minLoss / print_every}")
                    print(f"Validation loss: {devLoss}")

                    plot_loss_avg = plot_loss_total / plot_every
                    plotLosses.append((plot_loss_avg, i))
                    plot_loss_total = 0

                    plotDevLosses.append((devLoss, i))

    showPlot(*list(zip(*plotLosses)),
             f"Training losses for {datasetName} {'using' if curriculumLearning else 'without'} curriculum learning")
    showPlot(*list(zip(*plotDevLosses)),
             f"Training losses for {datasetName} {'using' if curriculumLearning else 'without'} curriculum learning")
    return optimalEncoder, optimalDecoder
