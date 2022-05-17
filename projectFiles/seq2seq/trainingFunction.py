from random import random

import torch
from torch import nn

from projectFiles.constants import device
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS
from projectFiles.seq2seq.lossFunction import maskNLLLoss


def train(inputIndices, lengths, outputIndices, mask, maxSimplifiedLength, encoder, decoder, decoderNoLayers,
          encoderOptimizer, decoderOptimizer, batchSize, clip, teacherForcingRatio):
    # Zero gradients
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()
    encoder.train()
    decoder.train()

    # Set device options
    inputIndices = inputIndices.to(device)
    outputIndices = outputIndices.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    noTotals = 0

    # Forward pass through encoder
    encoderOutputs, encoderHidden = encoder(inputIndices, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoderInput = torch.LongTensor([[SOS for _ in range(batchSize)]])
    decoderInput = decoderInput.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoderHidden = encoderHidden[:decoderNoLayers]

    # Determine if we are using teacher forcing this iteration
    useTeacherForcing = True if random() < teacherForcingRatio else False

    # Forward batch of sequences through decoder one time step at a time
    if useTeacherForcing:
        for t in range(maxSimplifiedLength):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )
            # Teacher forcing: next input is current target
            decoderInput = outputIndices[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoderOutput, outputIndices[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            noTotals += nTotal
    else:
        for t in range(maxSimplifiedLength):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoderOutput.topk(1)
            decoderInput = torch.LongTensor([[topi[i][0] for i in range(batchSize)]])
            decoderInput = decoderInput.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoderOutput, outputIndices[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            noTotals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoderOptimizer.step()
    decoderOptimizer.step()

    return sum(print_losses) / noTotals
