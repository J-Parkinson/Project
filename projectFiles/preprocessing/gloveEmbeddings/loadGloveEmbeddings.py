import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from projectFiles.constants import baseLoc
from projectFiles.seq2seq.constants import gloveWidth


def createGloveWord2VecFormat():
    glove2word2vec(glove_input_file=f"{baseLoc}/datasets/glove/glove.6B.300d.txt",
                   word2vec_output_file=f"{baseLoc}/datasets/glove/glove.6B.300d.gensim.txt")


def loadGloveAsWord2Vec():
    gloveModel = KeyedVectors.load_word2vec_format(f"{baseLoc}/datasets/glove/glove.6B.300d.gensim.txt", binary=False)
    return gloveModel


def getWordGlove(token, unknowns=False):
    if token in gloveModel.index_to_key:
        return gloveModel[token]
    if unknowns:
        return gloveModel["<unk>"]
    else:
        gloveModel[token] = 2 * np.random.beta(a=3, b=3, size=(
        gloveWidth,)) - 1  # https://leakyrelu.home.blog/2019/10/18/using-glove-word-embeddings-with-seq2seq-encoder-decoder-in-pytorch/
        # We adapt to ensure sampled values are in [-1, 1]
        return gloveModel[token]


def gloveEmbeddings(tokenized):
    return [getWordGlove(token) for token in tokenized]


gloveModel = loadGloveAsWord2Vec()
sizeOfEmbeddings = 300
