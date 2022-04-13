from random import random

import torch
from torch import optim, nn

from projectFiles.helpers.epochData import epochData
from projectFiles.helpers.epochTiming import Timer
from projectFiles.seq2seq.constants import device, SOS, teacher_forcing_ratio, EOS, maxLengthSentence
from projectFiles.seq2seq.embeddingLayers import embeddingLayer


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, embedding,
          max_length=maxLengthSentence):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # There are two 'embeddding' layers - one here, and one inside the encoder.
    # This is because for indices embeddings it needs to store learnt embeddings, whilst for Bert and Glove it uses
    # pretrained embeddings
    # However, BERT needs context for the entire sentence, hence this is done here outside the input loop.
    input_tensor_embedding_func = embeddingLayer(embedding, encoder.input_size, encoder.hidden_size)
    input_tensor_embedding = input_tensor_embedding_func(input_tensor)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor_embedding[ei], encoder_hidden)
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


def validationLoss(input_tensors, target_tensors_set, encoder, decoder, criterion, embedding,
                   max_length=maxLengthSentence):
    losses = []
    with torch.no_grad():
        for input_tensor, target_tensors in zip(input_tensors, target_tensors_set):
            for target_tensor in target_tensors:
                encoder_hidden = encoder.initHidden()

                input_length = input_tensor.size(0)
                target_length = target_tensor.size(0)

                encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                loss = 0

                # There are two 'embeddding' layers - one here, and one inside the encoder.
                # This is because for indices embeddings it needs to store learnt embeddings, whilst for Bert and Glove it uses
                # pretrained embeddings
                # However, BERT needs context for the entire sentence, hence this is done here outside the input loop.
                input_tensor_embedding_func = embeddingLayer(embedding, encoder.input_size, encoder.hidden_size)
                input_tensor_embedding = input_tensor_embedding_func(input_tensor)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_tensor_embedding[ei], encoder_hidden)
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


def trainMultipleIterations(trainingMetadata=None, encoder=None, decoder=None, allData=None, datasetName=None,
                            embedding=None, startIter=0):
    if encoder and decoder and allData and datasetName and embedding:
        trainingMetadata = epochData(encoder, decoder, allData, embedding, datasetName, startIter=startIter)
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
    criterion = nn.NLLLoss()

    iLocal = 0

    trainingData = trainingMetadata.data.train
    validationData = trainingMetadata.data.dev

    plot_loss_avg = devLoss = 0

    for m, pair in enumerate(trainingData):
        if isinstance(pair, tuple):
            targetTensorToUse = pair[1]
            pair = pair[0]
            target_tensors = [pair.allSimpleTorches[targetTensorToUse]]
        else:
            target_tensors = pair.allSimpleTorches
        input_tensor = pair.originalTorch

        j = 0

        while iLocal < trainingMetadata.startIter and j < len(target_tensors):
            iLocal += 1
            j += 1

        if j == len(target_tensors):
            continue

        for k, target_tensor in enumerate(target_tensors):
            iLocal += 1
            trainingMetadata.iGlobal += 1
            print(trainingMetadata.iGlobal)

            if trainingMetadata.checkIfEpochShouldEnd():
                trainingMetadata.epochFinished = False
                return trainingMetadata

            if k < j:
                continue

            loss = train(input_tensor, target_tensor, trainingMetadata.encoder, trainingMetadata.decoder,
                         encoder_optimizer, decoder_optimizer, criterion, trainingMetadata.embedding)
            print_loss_total += loss
            plot_loss_total += loss

            # We split print_every and plot_every here back into two
            # plot_every is controlled by iGlobal, print_every is controlled by iLocal

            if (trainingMetadata.iGlobal + 1) % trainingMetadata.valCheckEvery == 0:
                if not trainingMetadata.notFirstYet:
                    trainingMetadata.notFirstYet = True
                else:
                    print_loss_avg = print_loss_total / trainingMetadata.valCheckEvery
                    trainingMetadata.minLoss = min(trainingMetadata.minLoss, print_loss_total)

                    # Calculate validation loss
                    devLoss = validationLoss([dataVal.originalTorch for dataVal in validationData],
                                             [dataVal.allSimpleTorches for dataVal in validationData],
                                             trainingMetadata.encoder, trainingMetadata.decoder, criterion,
                                             trainingMetadata.embedding)
                    trainingMetadata.minDevLoss = min(trainingMetadata.minDevLoss, devLoss)

                    if trainingMetadata.minDevLoss == devLoss:
                        trainingMetadata.lastIterOfDevLossImp = trainingMetadata.iGlobal
                        trainingMetadata.optimalEncoder = trainingMetadata.encoder.state_dict()
                        trainingMetadata.optimalDecoder = trainingMetadata.decoder.state_dict()
                        with open(f"{trainingMetadata.fileSaveDir}/epochRun.txt", "w+") as file:
                            file.write(trainingMetadata.getAttributeStr(iLocal))
                        torch.save(trainingMetadata.optimalEncoder, f"{trainingMetadata.fileSaveDir}/encoder.pt")
                        torch.save(trainingMetadata.optimalDecoder, f"{trainingMetadata.fileSaveDir}/decoder.pt")

                    print_loss_total = 0

                    print("_____________________")
                    print(f"Iteration {iLocal + 1}")
                    timer.printTimeDiff()
                    timer.printTimeBetweenChecks()
                    print(f"Loss average: {print_loss_avg}")
                    print(
                        f"Min avg loss per {trainingMetadata.valCheckEvery} iterations: {trainingMetadata.minLoss / trainingMetadata.valCheckEvery}")
                    print(f"Validation loss: {devLoss}")

                    plot_loss_avg = plot_loss_total / trainingMetadata.valCheckEvery
                    trainingMetadata.plot_losses.append((trainingMetadata.iGlobal + 1, plot_loss_avg))
                    plot_loss_total = 0

                    trainingMetadata.plot_dev_losses.append((trainingMetadata.iGlobal + 1, devLoss))

    trainingMetadata.plot_losses.append((trainingMetadata.iGlobal + 1, plot_loss_avg))
    trainingMetadata.plot_dev_losses.append((trainingMetadata.iGlobal + 1, devLoss))

    return trainingMetadata
