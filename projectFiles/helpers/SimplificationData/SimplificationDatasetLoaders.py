import torch
from torch.utils.data import DataLoader

from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import model as BERTmodel
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import PAD
from projectFiles.seq2seq.constants import device


# Stores training, dev and test set splits as data loaders
# Handles batching as well as embeddings
# Embeddings are dealt with within the collate function to prevent memory issues with embeddings

class simplificationDatasetLoader():

    def convertToInputEmbedding(self, views, embedding):
        lengthSentences = views[0].maxSentenceLen
        noSentencesPad = self.batch_size - len(views)
        if embedding == embeddingType.bert:
            # padding at sentence level pre done by bert encoder
            # need to pad here though
            padding = [torch.tensor([[PAD] for _ in range(lengthSentences)], dtype=torch.int32, device=device) for _ in
                       range(noSentencesPad)]
            dataPadded = tuple([view.originalTorch for view in views] + padding)
            embeddings = BERTmodel.get_input_embeddings()(torch.hstack(dataPadded)).swapaxes(0, 1)
        else:
            # Trained embeddings within encoder instead
            padding = [torch.tensor([[PAD] for _ in range(lengthSentences)], dtype=torch.int32, device=device) for _ in
                       range(noSentencesPad)]
            embeddings = torch.stack([view.originalTorch for view in views] + padding)
        return embeddings

    def convertToOutputEmbedding(self, views, embedding):
        lengthSentences = views[0].maxSentenceLen
        noSentencesPad = self.batch_size - len(views)
        if embedding == embeddingType.bert:
            # padding at sentence level pre done by bert encoder
            # need to pad here though
            padding = [torch.tensor([[PAD] for _ in range(lengthSentences)], dtype=torch.int32, device=device) for _ in
                       range(noSentencesPad)]
            dataPadded = tuple([view.simpleTorch for view in views] + padding)
            embeddings = BERTmodel.get_input_embeddings()(torch.hstack(dataPadded)).swapaxes(0, 1)
        else:
            # Trained embeddings within encoder instead
            padding = [torch.tensor([[PAD] for _ in range(lengthSentences)], dtype=torch.int32, device=device) for _ in
                       range(noSentencesPad)]
            embeddings = torch.stack([view.simpleTorch for view in views] + padding)
        return embeddings

    def _collateFunction(self, views):
        inputEmbeddings = self.convertToInputEmbedding(views, self.embedding)
        outputEmbeddings = self.convertToOutputEmbedding(views, self.embedding)
        inputIndices = self.convertToInputEmbedding(views, embeddingType.indices)
        outputIndices = self.convertToOutputEmbedding(views, embeddingType.indices)
        return {"input": inputEmbeddings, "output": outputEmbeddings, "indicesInput": inputIndices,
                "indicesOutput": outputIndices}

    def __init__(self, simpDS, embedding, batch_size=128):
        self.dataset = simpDS.dataset
        self.embedding = embedding
        self.batch_size = batch_size
        self.trainDL = DataLoader(simpDS.train, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
        self.devDL = DataLoader(simpDS.dev, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
        self.testDL = DataLoader(simpDS.test, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
