# Luong attention layer
import torch
import torch.nn.functional as F
from torch import nn


class AttentionModel(nn.Module):
    def __init__(self, hiddenSize):
        super(AttentionModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.attn = nn.Linear(self.hiddenSize, hiddenSize)

    def forward(self, hidden, encoderOutput):
        energy = self.attn(encoderOutput)
        attn_energies = torch.sum(hidden * energy, dim=2)
        attn_energies = attn_energies.T

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
