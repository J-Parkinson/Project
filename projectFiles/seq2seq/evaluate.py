import torch

from projectFiles.helpers.embeddingType import embeddingType, convertDataBackToWords
from projectFiles.seq2seq.constants import device


# from projectFiles.seq2seq.deprecated.loadEncoderDecoder import loadEncoderDecoder, loadDataForEncoderDecoder

def evaluate(trainingMetadata):
    dataLoader = trainingMetadata.data.testDL
    encoder = trainingMetadata.encoder
    encoder.eval()
    decoder = trainingMetadata.decoder
    decoder.eval()
    batchSize = trainingMetadata.batchSize
    maxLen = trainingMetadata.maxLenSentence
    embedding = trainingMetadata.embedding

    allInputIndices = []
    allOutputIndices = []
    allDecoderOutputs = []

    with torch.no_grad():
        for batchNo, batch in enumerate(dataLoader):

            encoder_hidden = encoder.initHidden()

            input = batch["input"]
            indicesInput = batch["indicesInput"]
            allInputIndices.append(indicesInput)
            indicesOutput = batch["indicesOutput"]
            allOutputIndices.append(indicesOutput)

            encoder_outputs = torch.zeros(batchSize, maxLen, encoder.hidden_size, device=device)

            for token in range(maxLen):
                encoder_output, encoder_hidden = encoder(
                    input[:, token], encoder_hidden)
                encoder_outputs[:, token] = encoder_output[:, 0]

            decoder_input = input[:, 0]
            decoder_outputs = []
            decoder_hidden = encoder_hidden

            for di in range(1, maxLen):
                decoder_outputs.append(decoder_input)
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoded_output = decoder_output.squeeze()
                if embedding != embeddingType.bert:
                    _, decoded_output = decoded_output.topk(1)
                decoder_input = decoded_output.detach()  # detach from history as input

            decoder_outputs.append(decoder_input)

            decoder_outputs = torch.dstack(decoder_outputs)
            allDecoderOutputs.append(decoder_outputs)

    allPredicted = torch.vstack(allDecoderOutputs).detach().swapaxes(1, 2)
    allInputIndices = torch.vstack(allInputIndices).detach()
    allOutputIndices = torch.vstack(allOutputIndices).detach()

    allInputs, allOutputs, allPredicted = convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted,
                                                                 embedding,
                                                                 batchSize)

    allData = [{"input": inp, "output": out, "predicted": pred} for inp, out, pred in
               zip(allInputs, allOutputs, allPredicted)]

    return allData

# def evaluateOld(encoder, decoder, set, embedding, max_length=maxLengthSentence):
#    with torch.no_grad():
#        encoder_hidden = encoder.initHidden()
#
#        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#        # There are two 'embeddding' layers - one here, and one inside the encoder.
#        # This is because for indices embeddings it needs to store learnt embeddings, whilst for Bert and Glove it uses
#        # pretrained embeddings
#        # However, BERT needs context for the entire sentence, hence this is done here outside the input loop.
#        input_tensor_embedding_func = inputEmbeddingLayer(embedding, encoder.input_size, encoder.hidden_size)
#        input_tensor_embedding = input_tensor_embedding_func(set)
#
#        input_length = input_tensor_embedding.size(0)
#
#        for ei in range(input_length):
#            encoder_output, encoder_hidden = encoder(input_tensor_embedding[ei],
#                                                     encoder_hidden)
#            encoder_outputs[ei] += encoder_output[0, 0]
#
#        decoder_input = torch.tensor([[SOS]], device=device)  # SOS
#
#        decoder_hidden = encoder_hidden
#
#        decoded_words = []
#        decoder_attentions = torch.zeros(max_length, max_length)
#
#        for di in range(max_length):
#            decoder_output, decoder_hidden, decoder_attention = decoder(
#                decoder_input, decoder_hidden, encoder_outputs)
#            decoder_attentions[di] = decoder_attention.data
#            topv, topi = decoder_output.data.topk(1)
#            if topi.item() == EOS:
#                decoded_words.append('<EOS>')
#                break
#            else:
#                decoded_words.append(indicesReverseList[topi.item()])
#
#            decoder_input = topi.squeeze().detach()
#
#        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:

# def evaluateRandomly(encoder, decoder, dataset, embedding, n=5):
#    for i in range(n):
#        set = choice(dataset.test.dataset)
#        print()
#        print('-----------------------------------')
#        print('>', " ".join(set.originalTokenized))
#        for sentence in set.allSimpleTokenized:
#            print('=', " ".join(sentence))
#        output_words, attentions = evaluate(encoder, decoder, set, embedding)
#        output_sentence = ' '.join(output_words)
#        print('<', output_sentence)
