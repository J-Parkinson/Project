import torch

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS
from projectFiles.seq2seq.constants import device

bertSOS = list(
    model.get_input_embeddings()(torch.tensor([101], dtype=torch.long, device=device)).detach().cpu().numpy()[0])


def getDecoderInput(embedding, size):
    if embedding == embeddingType.bert:
        return torch.tensor([[bertSOS for _ in range(size)]], dtype=torch.float32, device=device)
    else:
        return torch.tensor([[SOS for _ in range(size)]], dtype=torch.long, device=device)
