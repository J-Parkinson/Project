import itertools

import numpy as np
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

    def convertToInputEmbeddingBERT(self, views):
        allOriginalTokenized, _1 = self._convertToInputEmbeddingPreprocessing(views)
        lengths = torch.tensor([len(view) for view in allOriginalTokenized], dtype=torch.int, device="cpu")
        allOriginalPadded = self._padSentences(allOriginalTokenized, "[PAD]")
        allOriginalPaddedTorch = torch.tensor(allOriginalPadded, device=device, dtype=torch.long)
        allOriginalPaddedSwap = allOriginalPadded.swapaxes(0, 1)
        allOriginalTorch = torch.tensor(allOriginalPaddedSwap, device=device, dtype=torch.long)
        embeddings = BERTmodel.get_input_embeddings()(allOriginalTorch).swapaxes(0, 1)
        return embeddings, lengths, allOriginalPaddedTorch

    def convertToInputEmbeddingIndicesGlove(self, views):
        allOriginalTokenized, allOriginalIndices = self._convertToInputEmbeddingPreprocessing(views)
        lengths = torch.tensor([len(view) for view in allOriginalTokenized], dtype=torch.int, device="cpu")
        allOriginalPadded = self._padSentences(allOriginalIndices, PAD)
        allOriginalTorch = torch.tensor(allOriginalPadded, device=device, dtype=torch.long)
        return allOriginalTorch, lengths

    def convertToOutputEmbeddingBERT(self, views):
        allSimpleTokenized, _1 = self._convertToOutputEmbeddingPreprocessing(views)
        maxSimplifiedLength = max([len(view.simpleTokenized) for view in views])
        allSimplePadded = self._padSentences(allSimpleTokenized, "[PAD]")
        mask = torch.tensor(allSimplePadded != "[PAD]", dtype=torch.bool)
        allSimplePaddedTorch = torch.tensor(allSimplePadded, device=device, dtype=torch.long)
        allSimplePaddedSwap = allSimplePadded.swapaxes(0, 1)
        allSimpleTorch = torch.tensor(allSimplePaddedSwap, device=device, dtype=torch.long)
        embeddings = BERTmodel.get_input_embeddings()(allSimpleTorch).swapaxes(0, 1)
        return embeddings, mask, maxSimplifiedLength, allSimplePaddedTorch

    def convertToOutputEmbeddingIndicesGlove(self, views):
        allSimpleTokenized, allSimpleIndices = self._convertToOutputEmbeddingPreprocessing(views)
        maxSimplifiedLength = max([len(view) for view in allSimpleTokenized])
        allSimplePadded = self._padSentences(allSimpleIndices, PAD)
        mask = torch.tensor(allSimplePadded != PAD, dtype=torch.bool, device=device)
        allSimpleTorch = torch.tensor(allSimplePadded, device=device, dtype=torch.long)
        return allSimpleTorch, mask, maxSimplifiedLength

    def _collateFunction(self, views):
        views.sort(key=lambda view: len(view.originalTokenized), reverse=True)
        if self.embedding == embeddingType.bert:
            inputEmbeddings, lengths, inputIndices = self.convertToInputEmbeddingBERT(views)
            outputEmbeddings, mask, maxSimplifiedLength, outputIndices = self.convertToOutputEmbeddingBERT(views)
        else:
            inputEmbeddings, lengths = self.convertToInputEmbeddingIndicesGlove(views)
            inputIndices = inputEmbeddings
            outputEmbeddings, mask, maxSimplifiedLength = self.convertToOutputEmbeddingIndicesGlove(views)
            outputIndices = outputEmbeddings

        return {"inputEmbeddings": inputEmbeddings,
                "outputEmbeddings": outputEmbeddings,
                "inputIndices": inputIndices,
                "outputIndices": outputIndices,
                "lengths": lengths,
                "maxSimplifiedLength": maxSimplifiedLength,
                "mask": mask}

    def __init__(self, simpDS, embedding, batch_size=128):
        self.dataset = simpDS.dataset
        self.embedding = embedding
        self.batch_size = batch_size
        self.trainDL = DataLoader(simpDS.train, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
        self.devDL = DataLoader(simpDS.dev, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
        self.testDL = DataLoader(simpDS.test, batch_size=batch_size, collate_fn=self._collateFunction, shuffle=False)
