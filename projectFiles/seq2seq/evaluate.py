from random import choice

import torch

from projectFiles.seq2seq.constants import EOS, SOS, device, maxLengthSentence, indicesRaw
# from projectFiles.seq2seq.deprecated.loadEncoderDecoder import loadEncoderDecoder, loadDataForEncoderDecoder
from projectFiles.seq2seq.embeddingLayers import inputEmbeddingLayer


def evaluate(encoder, decoder, set, embedding, max_length=maxLengthSentence):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # There are two 'embeddding' layers - one here, and one inside the encoder.
        # This is because for indices embeddings it needs to store learnt embeddings, whilst for Bert and Glove it uses
        # pretrained embeddings
        # However, BERT needs context for the entire sentence, hence this is done here outside the input loop.
        input_tensor_embedding_func = inputEmbeddingLayer(embedding, encoder.input_size, encoder.hidden_size)
        input_tensor_embedding = input_tensor_embedding_func(set)

        input_length = input_tensor_embedding.size(0)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor_embedding[ei],
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

def evaluateRandomly(encoder, decoder, dataset, embedding, n=5):
    for i in range(n):
        set = choice(dataset.test.dataset)
        print()
        print('-----------------------------------')
        print('>', " ".join(set.originalTokenized))
        for sentence in set.allSimpleTokenized:
            print('=', " ".join(sentence))
        output_words, attentions = evaluate(encoder, decoder, set, embedding)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)


def evaluateAllEpoch(epochData):
    testData = epochData.data.test.dataset
    for set in testData:
        output_words, attentions = evaluate(epochData.encoder, epochData.decoder, set, epochData.embedding)
        output_sentence = ' '.join(output_words)
        set.addPredicted(output_sentence)
    return epochData


def evaluateAll(encoder, decoder, dataset, embedding):
    testData = dataset.test.dataset
    for set in testData:
        output_words, attentions = evaluate(encoder, decoder, set, embedding)
        output_sentence = ' '.join(output_words)
        set.addPredicted(output_sentence)
    return testData

# def loadEncoderDecoderDatasetAndEvaluateRandomly(filepath, hiddenLayerWidth=256, maxIndices=253401):
#    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)
#
#    _, datasetData, _, datasetName = loadDataForEncoderDecoder(filepath, maxIndices)
#
#    evaluateRandomly(encoder, decoder, datasetData)
#
#
# def loadEncoderDecoderDatasetAndEvaluateAll(filepath, hiddenLayerWidth=256, maxIndices=253401):
#    encoder, decoder = loadEncoderDecoder(filepath, hiddenLayerWidth, maxIndices)
#
#    _, datasetData, _, datasetName = loadDataForEncoderDecoder(filepath, maxIndices)
#
#    return evaluateAll(encoder, decoder, datasetData)

# loadEncoderDecoderDatasetAndEvaluateAll("seq2seq/trainedModels/optimal_asset_025043")
