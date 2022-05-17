import torch
from torch import nn as nn

from projectFiles.constants import device
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.gloveEmbeddings.gloveNetwork import GloveEmbeddings


class EncoderRNN(nn.Module):

    def __init__(self, hiddenSize, embeddingTokenSize, embedding, noLayers=2, dropout=0.1):
        # print(input_size)
        super(EncoderRNN, self).__init__()
        self.n_layers = noLayers
        self.hiddenSize = hiddenSize
        self.embeddingTokenSize = embeddingTokenSize
        self.embedding = embedding

        self.gru = nn.GRU(hiddenSize, hiddenSize, noLayers,
                          dropout=(0 if noLayers == 1 else dropout), bidirectional=True)

        if embedding == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hiddenSize)
        elif embedding == embeddingType.glove:
            self.embeddingLayer = GloveEmbeddings(embeddingTokenSize)
        else:
            self.embeddingLayer = lambda x: x

    def forward(self, input, inputLengths, hidden=None):
        embedded = self.embeddingLayer(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, self.batchSize, self.hiddenSize, device=device)
