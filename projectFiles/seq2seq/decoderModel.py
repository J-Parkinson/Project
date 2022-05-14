import torch
import torch.nn as nn
import torch.nn.functional as F

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.gloveEmbeddings.gloveNetwork import GloveEmbeddings
from projectFiles.seq2seq.attentionModel import AttentionModel
from projectFiles.seq2seq.constants import maxLengthSentence


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

class AttnDecoderRNN(nn.Module):
    def __init__(self, hiddenSize, embeddingTokenSize, embeddingVersion, noLayers=2, dropout=0.1,
                 maxLength=maxLengthSentence):
        super(AttnDecoderRNN, self).__init__()

        self.hiddenSize = hiddenSize
        self.embeddingTokenSize = embeddingTokenSize
        self.embeddingVersion = embeddingVersion
        self.dropout = dropout
        self.maxLength = maxLength
        self.noLayers = noLayers

        if embeddingVersion == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hiddenSize)
        elif embeddingVersion == embeddingType.glove:
            self.embeddingLayer = GloveEmbeddings(embeddingTokenSize)
        else:
            self.embeddingLayer = lambda x: x

        self.attention = AttentionModel(hiddenSize)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.gru = nn.GRU(hiddenSize, hiddenSize, self.noLayers, dropout=(0 if self.noLayers == 1 else dropout))
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.embeddingTokenSize)

    def forward(self, input, hidden, encoderOutputs):
        # This is ran one word at a time rather than for the entire batch
        embedding = self.embeddingLayer(input)
        embedding = self.dropoutLayer(embedding)
        gruOutput, hidden = self.gru(embedding, hidden)
        attentionWeights = self.attention(gruOutput, encoderOutputs)
        context = attentionWeights.bmm(encoderOutputs.transpose(0, 1))
        gruOutput = gruOutput.squeeze(0)
        context = context.squeeze(1)
        concatGruContext = torch.cat((gruOutput, context), 1)
        output = torch.tanh(self.concat(concatGruContext))
        if self.embeddingVersion != embeddingType.bert:
            output = F.softmax(self.out(output), dim=1)
        return output, hidden, attentionWeights
