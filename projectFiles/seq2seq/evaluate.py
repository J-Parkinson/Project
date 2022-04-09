from random import choice

import torch

from projectFiles.seq2seq.constants import EOS, SOS, device, maxLengthSentence, indicesRaw
from projectFiles.seq2seq.loadEncoderDecoder import loadEncoderDecoder, loadDataForEncoderDecoder


def evaluate(encoder, decoder, input_tensor, max_length=maxLengthSentence):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(indicesRaw[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:

def evaluateRandomly(encoder, decoder, dataset, n=5):
    for i in range(n):
        set = choice(dataset.test.dataset)
        print()
        print('-----------------------------------')
        print('>', " ".join(set.originalTokenized))
        for sentence in set.allSimpleTokenized:
            print('=', " ".join(sentence))
        output_words, attentions = evaluate(encoder, decoder, set.originalTorch)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)


def evaluateAllEpoch(epochData):
    testData = epochData.data.test.dataset
    for set in testData:
        output_words, attentions = evaluate(epochData.encoder, epochData.decoder, set.originalTorch)
        output_sentence = ' '.join(output_words)
        set.addPredicted(output_sentence)
    return epochData


def evaluateAll(encoder, decoder, dataset):
    testData = dataset.test.dataset
    for set in testData:
        output_words, attentions = evaluate(encoder, decoder, set.originalTorch)
        output_sentence = ' '.join(output_words)
        set.addPredicted(output_sentence)
    return testData


def loadEncoderDecoderDatasetAndEvaluateRandomly(filepath, hiddenLayerWidth=256, maxIndices=253401):
    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)

    _, datasetData, _, datasetName = loadDataForEncoderDecoder(filepath, maxIndices)

    evaluateRandomly(encoder, decoder, datasetData)


def loadEncoderDecoderDatasetAndEvaluateAll(filepath, hiddenLayerWidth=256, maxIndices=253401):
    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)

    _, datasetData, _, datasetName = loadDataForEncoderDecoder(filepath, maxIndices)

    return evaluateAll(encoder, decoder, datasetData)

# loadEncoderDecoderDatasetAndEvaluateAll("seq2seq/trainedModels/optimal_asset_025043")
