import torch
from transformers.utils import PaddingStrategy

from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import tokenizer
# Note that S0S == [PAD] in BERT tokenizer, but we never pad sentences so this is reused from the NLTK code as is
from projectFiles.seq2seq.constants import device


class simplificationSetBERT(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self.tokenise()

    def _addPadding(self, sentence):
        return sentence + ["[PAD]" for _ in range(self.maxSentenceLen - len(sentence))]

    def tokenise(self):
        self.originalTokenized = tokenizer.tokenize(self.originalTokenized)
        self.allSimpleTokenized = [tokenizer.tokenize(sentence) for sentence in self.allSimpleTokenized]
        self.originalTokenizedPadded = self._addPadding(self.originalTokenized)
        self.allSimpleTokenizedPadded = [self._addPadding(sentence) for sentence in self.allSimpleTokenized]

    def addIndices(self):
        self.originalIndices = tokenizer.encode_plus(self.original,
                                                     return_attention_mask=False,
                                                     return_token_type_ids=False,
                                                     add_special_tokens=True,
                                                     max_length=self.maxSentenceLen,
                                                     padding=PaddingStrategy.MAX_LENGTH)["input_ids"]
        self.allSimpleIndices = [tokenizer.encode_plus(sentence,
                                                       return_attention_mask=False,
                                                       return_token_type_ids=False,
                                                       add_special_tokens=True,
                                                       max_length=self.maxSentenceLen,
                                                       padding=PaddingStrategy.MAX_LENGTH)["input_ids"]
                                 for sentence in self.allSimple]

    # Creates torch tensors from each sentence indices - for use by Cuda (`device'-depending)
    def torchSet(self):
        self.originalTorch = torch.tensor(self.originalIndices, dtype=torch.long, device=device).view(-1, 1)
        self.allSimpleTorches = [torch.tensor(simpleIndex, dtype=torch.long, device=device).view(-1, 1) for simpleIndex
                                 in self.allSimpleIndices]

        return (self.originalTorch, self.allSimpleTorches)
