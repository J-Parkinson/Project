import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from projectFiles.constants import device
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import PAD


# Stores training, dev and test set splits as data loaders
# Handles batching as well as embeddings
# Embeddings are dealt with within the collate function to prevent memory issues with embeddings

class simplificationDatasetLoader():

    # Here are the NEW steps for this loader to handle
    # - Data is pre tokenized so we need to add padding
    # - We also need to convert to torch tensor
    # - We also need to create masks
    # - We also need to create length list (to CPU)
    # - We also save the max length of sentences

    def _padSentences(self, sentencesTokenized, padVal):
        return np.array(list(itertools.zip_longest(*sentencesTokenized, fillvalue=padVal)))

    def _convertToInputEmbeddingPreprocessing(self, views):
        allOriginalTokenized = [view.originalTokenized for view in views]
        allOriginalIndices = [view.originalIndices for view in views]
        return allOriginalTokenized, allOriginalIndices

    def _convertToOutputEmbeddingPreprocessing(self, views):
        allSimpleTokenized = [view.simpleTokenized for view in views]
        allSimpleIndices = [view.simpleIndices for view in views]
        return allSimpleTokenized, allSimpleIndices

    def convertToInputEmbeddingIndicesGlove(self, views):
        allOriginalTokenized, allOriginalIndices = self._convertToInputEmbeddingPreprocessing(views)
        lengths = torch.tensor([len(view) for view in allOriginalTokenized], dtype=torch.int, device="cpu")
        allOriginalPadded = self._padSentences(allOriginalIndices, PAD)
        allOriginalTorch = torch.tensor(allOriginalPadded, device=device, dtype=torch.long)
        return allOriginalTorch, lengths

    def convertToOutputEmbeddingIndicesGlove(self, views):
        allSimpleTokenized, allSimpleIndices = self._convertToOutputEmbeddingPreprocessing(views)
        maxSimplifiedLength = max([len(view) for view in allSimpleTokenized])
        allSimplePadded = self._padSentences(allSimpleIndices, PAD)
        mask = torch.tensor(allSimplePadded != PAD, dtype=torch.bool, device=device)
        allSimpleTorch = torch.tensor(allSimplePadded, device=device, dtype=torch.long)
        return allSimpleTorch, mask, maxSimplifiedLength

    def _collateFunction(self, views):
        views.sort(key=lambda view: len(view.originalTokenized), reverse=True)
        inputEmbeddings, lengths = self.convertToInputEmbeddingIndicesGlove(views)
        outputEmbeddings, mask, maxSimplifiedLength = self.convertToOutputEmbeddingIndicesGlove(views)

        return {"inputEmbeddings": inputEmbeddings,
                "outputEmbeddings": outputEmbeddings,
                "lengths": lengths,
                "maxSimplifiedLength": maxSimplifiedLength,
                "mask": mask}

    def __init__(self, simpDS, embedding, batch_size=128):
        self.dataset = simpDS.dataset
        self.embedding = embedding
        self.batch_size = batch_size
        self.trainDL = DataLoader(simpDS.train, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False,
                                  drop_last=True)
        self.devDL = DataLoader(simpDS.dev, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False,
                                drop_last=True)
        self.testDL = DataLoader(simpDS.test, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False,
                                 drop_last=True)
