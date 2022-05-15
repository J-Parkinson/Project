import torch

from projectFiles.helpers.embeddingType import embeddingType, convertDataBackToWords
from projectFiles.seq2seq.constants import device


# from projectFiles.seq2seq.deprecated.loadEncoderDecoder import loadEncoderDecoder, loadDataForEncoderDecoder

def evaluate(trainingMetadata):
    print("Evaluation ---------------------------------------")
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

            encoderHidden = encoder.initHidden()

            input = batch["input"]
            indicesInput = batch["indicesInput"]
            allInputIndices.append(indicesInput)
            indicesOutput = batch["indicesOutput"]
            allOutputIndices.append(indicesOutput)

            encoderOutputs = torch.zeros(batchSize, maxLen, encoder.hiddenSize, device=device)

            for token in range(maxLen):
                encoderOutput, encoderHidden = encoder(
                    input[:, token], encoderHidden)
                encoderOutputs[:, token] = encoderOutput[:, 0]

            decoderInput = input[:, 0]
            decoderOutputs = []
            decoderHidden = encoderHidden

            for di in range(1, maxLen):
                decoderOutputs.append(decoderInput)
                decoderOutput, decoderHidden, decoderAttention = decoder(
                    decoderInput, decoderHidden, encoderOutputs)
                decodedOutput = decoderOutput.squeeze()
                if embedding != embeddingType.bert:
                    _, decodedOutput = decodedOutput.topk(1)
                decoderInput = decodedOutput.detach()  # detach from history as input

            decoderOutputs.append(decoderInput)

            decoderOutputs = torch.dstack(decoderOutputs)
            allDecoderOutputs.append(decoderOutputs)

    allPredicted = torch.vstack(allDecoderOutputs).detach().swapaxes(1, 2)
    allInputIndices = torch.vstack(allInputIndices).detach()
    allOutputIndices = torch.vstack(allOutputIndices).detach()

    allInputs, allOutputs, allPredicted = convertDataBackToWords(allInputIndices, allOutputIndices, allPredicted,
                                                                 embedding,
                                                                 batchSize)

    allData = [{"input": inp, "output": out, "predicted": pred} for inp, out, pred in
               zip(allInputs, allOutputs, allPredicted)]

    return allData

# def evaluateOld(encoder, decoder, set, embedding, maxLength=maxLengthSentence):
#    with torch.no_grad():
#        encoderHidden = encoder.initHidden()
#
#        encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize, device=device)
#
#        # There are two 'embeddding' layers - one here, and one inside the encoder.
#        # This is because for indices embeddings it needs to store learnt embeddings, whilst for Bert and Glove it uses
#        # pretrained embeddings
#        # However, BERT needs context for the entire sentence, hence this is done here outside the input loop.
#        inputTensor_embedding_func = inputEmbeddingLayer(embedding, encoder.input_size, encoder.hiddenSize)
#        inputTensor_embedding = inputTensor_embedding_func(set)
#
#        input_length = inputTensor_embedding.size(0)
#
#        for ei in range(input_length):
#            encoderOutput, encoderHidden = encoder(inputTensor_embedding[ei],
#                                                     encoderHidden)
#            encoderOutputs[ei] += encoderOutput[0, 0]
#
#        decoderInput = torch.tensor([[SOS]], device=device)  # SOS
#
#        decoderHidden = encoderHidden
#
#        decoded_words = []
#        decoderAttentions = torch.zeros(maxLength, maxLength)
#
#        for di in range(maxLength):
#            decoderOutput, decoderHidden, decoderAttention = decoder(
#                decoderInput, decoderHidden, encoderOutputs)
#            decoderAttentions[di] = decoderAttention.data
#            topv, topi = decoderOutput.data.topk(1)
#            if topi.item() == EOS:
#                decoded_words.append('<EOS>')
#                break
#            else:
#                decoded_words.append(indicesReverseList[topi.item()])
#
#            decoderInput = topi.squeeze().detach()
#
#        return decoded_words, decoderAttentions[:di + 1]


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
