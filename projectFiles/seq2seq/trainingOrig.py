from random import random

import torch
from torch import optim, nn

from projectFiles.helpers.epochTiming import Timer
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS, EOS
from projectFiles.seq2seq.constants import device, teacher_forcing_ratio, maxLengthSentence
from projectFiles.seq2seq.plots import showPlot


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=maxLengthSentence):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def validationLoss(input_tensors, target_tensors_set, encoder, decoder, criterion, max_length=maxLengthSentence):
    losses = []
    with torch.no_grad():
        for input_tensor, target_tensors in zip(input_tensors, target_tensors_set):
            for target_tensor in target_tensors:
                encoder_hidden = encoder.initHidden()

                input_length = input_tensor.size(0)
                target_length = target_tensor.size(0)

                encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                loss = 0

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_tensor[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_input = torch.tensor([[SOS]], device=device)

                decoder_hidden = encoder_hidden

                # We do not use teacher forcing during evaluation
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == EOS:
                        break
                losses.append(loss.item() / target_length)
    return sum(losses) / len(losses)


def trainIters(encoder, decoder, allData, datasetName, locationToSaveTo="trainedModels/", print_every=150,
               plot_every=100, learning_rate=0.01, startIter=0):
    timer = Timer()
    plot_losses = []
    plot_dev_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
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
        input_tensor = pair.originalTorch
        target_tensors = pair.allSimpleTorches
        j = 0
        l = 0
        while i < startIter and j < len(target_tensors):
            i += 1
            j += 1
        if j == len(target_tensors):
            continue

        for k, target_tensor in enumerate(target_tensors):
            i += 1
            if i > 300:
                continue
            print(i)

            if k < j:
                continue

            l += len(target_tensors)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if (i + 1) % print_every == 0:
                if not notFirstYet:
                    notFirstYet = True
                else:
                    print_loss_avg = print_loss_total / print_every
                    minLoss = min(minLoss, print_loss_total)

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

                    print_loss_total = 0

                    print("_____________________")
                    print(f"Iteration {i + 1}")
                    timer.printTimeDiff()
                    timer.printTimeBetweenChecks()
                    timer.calculateTimeLeftToRun(i + 1, len(trainingData) * l / m)
                    print(f"Loss average: {print_loss_avg}")
                    print(f"Min avg loss per {print_every} iterations: {minLoss / print_every}")
                    print(f"Validation loss: {devLoss}")

                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append((plot_loss_avg, i))
                    plot_loss_total = 0

                    plot_dev_losses.append((devLoss, i))

    showPlot(*list(zip(*plot_losses)),
             f"Training losses for {datasetName} {'using' if curriculumLearning else 'without'} curriculum learning")
    showPlot(*list(zip(*plot_dev_losses)),
             f"Training losses for {datasetName} {'using' if curriculumLearning else 'without'} curriculum learning")
    return optimalEncoder, optimalDecoder
