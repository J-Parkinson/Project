import torch
import torch.nn as nn
import torch.nn.functional as F

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.gloveEmbeddings.gloveNetwork import getGloveEmbeddingNN
from projectFiles.seq2seq.constants import device, maxLengthSentence


# NEED TO DEAL WITH [EOS] IN ENCODER

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#self.embedding = nn.Embedding(input_size, hidden_size)
class EncoderRNN(nn.Module):
    def __init__(self, embeddingTokenSize, hidden_size, embedding, batchSize):
        # print(input_size)
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embeddingTokenSize = embeddingTokenSize
        self.batchSize = batchSize
        self.embedding = embedding

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        if embedding == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hidden_size)
        elif embedding == embeddingType.glove:
            self.embeddingLayer = getGloveEmbeddingNN(self.embeddingTokenSize, self.hidden_size)
        else:
            self.embeddingLayer = lambda x: x

    def forward(self, input, hidden):
        embedded = self.embeddingLayer(input)
        embedded = embedded.unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batchSize, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embeddingTokenSize, embedding, batchSize, dropout=0.1,
                 max_length=maxLengthSentence):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embeddingTokenSize = embeddingTokenSize
        self.dropout_p = dropout
        self.max_length = max_length
        self.embedding = embedding
        self.batchSize = batchSize

        if embedding == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hidden_size)
        elif embedding == embeddingType.glove:
            self.embeddingLayer = getGloveEmbeddingNN(self.embeddingTokenSize, self.hidden_size)
        else:
            self.embeddingLayer = lambda x: x

        self.sigmoid = nn.Sigmoid()

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.embeddingTokenSize)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embeddingLayer(input)
        embedded = embedded.unsqueeze(1)

        # if self.embedding != embeddingType.indices:
        #    embedded = nn.Tanh()(embedded)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden.swapaxes(0, 1)), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)  # torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        if self.embedding != embeddingType.bert:
            output = F.log_softmax(self.out(output), dim=2)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)