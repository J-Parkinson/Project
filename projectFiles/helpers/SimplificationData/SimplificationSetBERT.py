from projectFiles.helpers.SimplificationData import simplificationSet
from projectFiles.preprocessing.bertEmbeddings.loadBertEmbeddingsModel import tokenizer


# Note that S0S == [PAD] in BERT tokenizer, but we never pad sentences so this is reused from the NLTK code as is


class simplificationSetBERT(simplificationSet):
    def __init__(self, original, allSimple, dataset, language="en"):
        super().__init__(original, allSimple, dataset, language)
        self.tokenise()

    def tokenise(self):
        self.originalTokenized = f"[CLS] {self.originalTokenized} [SEP]"
        self.allSimpleTokenized = [f"[CLS] {sentence} [SEP]" for sentence in self.allSimpleTokenized]
        self.originalTokenized = tokenizer.tokenize(self.originalTokenized)
        self.allSimpleTokenized = [tokenizer.tokenize(sentence) for sentence in self.allSimpleTokenized]

    def addIndices(self):
        self.originalIndices = tokenizer.convert_tokens_to_ids(self.originalTokenized)
        self.allSimpleIndices = [tokenizer.convert_tokens_to_ids(sentence) for sentence in self.allSimpleTokenized]
