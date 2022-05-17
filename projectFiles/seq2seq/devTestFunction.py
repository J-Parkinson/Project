import torch

from projectFiles.constants import device
from projectFiles.evaluation.easse.calculateEASSE import computeValidation
from projectFiles.helpers.embeddingType import convertDataBackToWords
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS
from projectFiles.seq2seq.lossFunction import maskNLLLoss


def validationMultipleBatches(batches, encoder, decoder, decoderNoLayers, batchSize):
    allDecoderOutputs = []
    allInputEmbeddings = []
    allOutputEmbeddings = []
    allLosses = 0

    for batchNo, batch in enumerate(batches):
        inputEmbeddings = batch["inputEmbeddings"]
        outputEmbeddings = batch["outputEmbeddings"]
        allInputEmbeddings.append(inputEmbeddings)
        allOutputEmbeddings.append(outputEmbeddings)
        loss, inputEmbeddings, outputEmbeddings, decoderOutputs = \
            validationEvaluationOneBatch(batch, encoder, decoder, decoderNoLayers, batchSize)
        allDecoderOutputs.append(decoderOutputs)
        allLosses += loss

    allPredicted = [sentence.cpu().detach().numpy() for batch in allDecoderOutputs for sentence in batch]
    allInputEmbeddings = [batch.swapaxes(0, 1) for batch in allInputEmbeddings]
    allInputEmbeddings = [sentence.cpu().detach().numpy() for batch in allInputEmbeddings for sentence in batch]
    allOutputEmbeddings = [batch.swapaxes(0, 1) for batch in allOutputEmbeddings]
    allOutputEmbeddings = [sentence.cpu().detach().numpy() for batch in allOutputEmbeddings for sentence in batch]

    allInputs, allOutputs, allPredicted = convertDataBackToWords(allInputEmbeddings, allOutputEmbeddings, allPredicted)

    results = computeValidation(allInputs, allOutputs, allPredicted)

    print("Validation calculated-----------------------------")

    return allLosses, results


def evaluationMultipleBatches(batches, encoder, decoder, decoderNoLayers, batchSize):
    print("Evaluation started")
    allDecoderOutputs = []
    allInputEmbeddings = []
    allOutputEmbeddings = []

    for batchNo, batch in enumerate(batches):
        inputEmbeddings = batch["inputEmbeddings"]
        outputEmbeddings = batch["outputEmbeddings"]
        allInputEmbeddings.append(inputEmbeddings)
        allOutputEmbeddings.append(outputEmbeddings)
        _, inputEmbeddings, outputEmbeddings, decoderOutputs = \
            validationEvaluationOneBatch(batch, encoder, decoder, decoderNoLayers, batchSize)
        allDecoderOutputs.append(decoderOutputs)

    allPredicted = [sentence.cpu().detach().numpy() for batch in allDecoderOutputs for sentence in batch]
    allInputEmbeddings = [batch.swapaxes(0, 1) for batch in allInputEmbeddings]
    allInputEmbeddings = [sentence.cpu().detach().numpy() for batch in allInputEmbeddings for sentence in batch]
    allOutputEmbeddings = [batch.swapaxes(0, 1) for batch in allOutputEmbeddings]
    allOutputEmbeddings = [sentence.cpu().detach().numpy() for batch in allOutputEmbeddings for sentence in batch]

    allInputs, allOutputs, allPredicted = convertDataBackToWords(allInputEmbeddings, allOutputEmbeddings, allPredicted)

    results = computeValidation(allInputs, allOutputs, allPredicted)

    allData = [{"input": inp, "outputs": out, "predicted": pred} for inp, out, pred in
               zip(allInputs, allOutputs, allPredicted)]

    print("Evaluation completed")

    return allData, results


def validationEvaluationOneBatch(batch, encoder, decoder, decoderNoLayers, batchSize):
    inputEmbeddings = batch["inputEmbeddings"]
    outputEmbeddings = batch["outputEmbeddings"]
    lengths = batch["lengths"]
    maxSimplifiedLength = batch["maxSimplifiedLength"]
    mask = batch["mask"]

    encoder.eval()
    decoder.eval()

    # Set device options
    inputEmbeddings = inputEmbeddings.to(device)
    outputEmbeddings = outputEmbeddings.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    noTotals = 0
    weightedLoss = 0

    with torch.no_grad():
        # Forward pass through encoder
        encoderOutputs, encoderHidden = encoder(inputEmbeddings, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoderOutputs = []
        decoderInput = torch.LongTensor([[SOS for _ in range(batchSize)]])
        decoderInput = decoderInput.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoderHidden = encoderHidden[:decoderNoLayers]

        for t in range(maxSimplifiedLength):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoderOutput.topk(1)
            decoderInput = torch.LongTensor([[topi[i][0] for i in range(batchSize)]])
            decoderInput = decoderInput.to(device)
            decoderOutputs.append(decoderInput)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoderOutput, outputEmbeddings[t], mask[t])
            loss += mask_loss
            weightedLoss += mask_loss.item() * nTotal
            noTotals += nTotal

    decoderOutputs = torch.vstack(decoderOutputs).swapaxes(0, 1)

    return weightedLoss / noTotals, inputEmbeddings, outputEmbeddings, decoderOutputs
