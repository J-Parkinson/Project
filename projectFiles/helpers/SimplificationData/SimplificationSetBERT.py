from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import tokenizer
# Note that S0S == [PAD] in BERT tokenizer, but we never pad sentences so this is reused from the NLTK code as is


class simplificationSetBERT(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self.tokenise()
        self.addIndices()

    def tokenise(self):
        self.originalTokenized = tokenizer.tokenize(self.originalTokenized)
        self.allSimpleTokenized = [tokenizer.tokenize(sentence) for sentence in self.allSimpleTokenized]

    def addIndices(self):
        self.originalIndices = tokenizer.encode_plus(self.original,
                                                     return_attention_mask=False,
                                                     return_token_type_ids=False,
                                                     add_special_tokens=True)["input_ids"]
        self.allSimpleIndices = [tokenizer.encode_plus(sentence,
                                                       return_attention_mask=False,
                                                       return_token_type_ids=False,
                                                       add_special_tokens=True)["input_ids"]
                                 for sentence in self.allSimple]
