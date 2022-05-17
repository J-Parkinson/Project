import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.SimplificationData.SimplificationDatasetLoaders import simplificationDatasetLoader
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningMetadata, curriculumLearningFlag
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.helpers.getHiddenSize import getHiddenSize
from projectFiles.helpers.getMaxLens import getMaxLens
from projectFiles.preprocessing.convertToPyTorch.simplificationDataToPyTorch import simplificationDataToPyTorch
from projectFiles.preprocessing.gloveEmbeddings.gloveNetwork import GloveEmbeddings
from projectFiles.preprocessing.indicesEmbeddings.loadIndexEmbeddings import SOS, indicesReverseList

USE_CUDA = torch.cuda.is_available()  #
device = torch.device("cuda" if USE_CUDA else "cpu")  #

embedding = embeddingType.indices  #
clMD = curriculumLearningMetadata(curriculumLearningFlag.randomized)  #
hiddenLayerWidthForIndices = 512  #
restrict = 200000000  #
batchSize = 64  #
batchesBetweenValidation = 50  #
minNoOccurencesForToken = 2  #
hiddenSize = getHiddenSize(embedding, hiddenLayerWidthForIndices)  #

maxLenSentence = getMaxLens(datasetToLoad.wikilarge, restrict=restrict)  #

# Also restricts length of max len sentence in each set (1-n and 1-1)
datasetLoaded = simplificationDataToPyTorch(datasetToLoad.wikilarge, embedding, clMD, maxLen=maxLenSentence,
                                            minOccurences=minNoOccurencesForToken)  #
print("Dataset loaded")
# batching
datasetBatches = simplificationDatasetLoader(datasetLoaded, embedding, batch_size=batchSize)  #

embeddingTokenSize = len(indicesReverseList)  #


######################################################################
# Define Models
# -------------
#
# Seq2Seq Model
# ~~~~~~~~~~~~~
#
# The brains of our chatbot is a sequence-to-sequence (seq2seq) model. The
# goal of a seq2seq model is to take a variable-length sequence as an
# input, and return a variable-length sequence as an output using a
# fixed-sized model.
#
# `Sutskever et al. <https://arxiv.org/abs/1409.3215>`__ discovered that
# by using two separate recurrent neural nets together, we can accomplish
# this task. One RNN acts as an **encoder**, which encodes a variable
# length input sequence to a fixed-length context vector. In theory, this
# context vector (the final hidden layer of the RNN) will contain semantic
# information about the query sentence that is input to the bot. The
# second RNN is a **decoder**, which takes an input word and the context
# vector, and returns a guess for the next word in the sequence and a
# hidden state to use in the next iteration.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
# Image source:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#


######################################################################
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# At the heart of our encoder is a multi-layered Gated Recurrent Unit,
# invented by `Cho et al. <https://arxiv.org/pdf/1406.1078v3.pdf>`__ in
# 2014. We will use a bidirectional variant of the GRU, meaning that there
# are essentially two independent RNNs: one that is fed the input sequence
# in normal sequential order, and one that is fed the input sequence in
# reverse order. The outputs of each network are summed at each time step.
# Using a bidirectional GRU will give us the advantage of encoding both
# past and future contexts.
#
# Bidirectional RNN:
#
# .. figure:: /_static/img/chatbot/RNN-bidirectional.png
#    :width: 70%
#    :align: center
#    :alt: rnn_bidir
#
# Image source: https://colah.github.io/posts/2015-09-NN-Types-FP/
#
# Note that an ``embedding`` layer is used to encode our word indices in
# an arbitrarily sized feature space. For our models, this layer will map
# each word to a feature space of size *hiddenSize*. When trained, these
# values should encode semantic similarity between similar meaning words.
#
# Finally, if passing a padded batch of sequences to an RNN module, we
# must pack and unpack padding around the RNN pass using
# ``nn.utils.rnn.pack_padded_sequence`` and
# ``nn.utils.rnn.pad_packed_sequence`` respectively.
#
# **Computation Graph:**
#
#    1) Convert word indexes to embeddings.
#    2) Pack padded batch of sequences for RNN module.
#    3) Forward pass through GRU.
#    4) Unpack padding.
#    5) Sum bidirectional GRU outputs.
#    6) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_seq``: batch of input sentences; shape=\ *(maxLength,
#    batch_size)*
# -  ``input_lengths``: list of sentence lengths corresponding to each
#    sentence in the batch; shape=\ *(batch_size)*
# -  ``hidden``: hidden state; shape=\ *(n_layers x num_directions,
#    batch_size, hiddenSize)*
#
# **Outputs:**
#
# -  ``outputs``: output features from the last hidden layer of the GRU
#    (sum of bidirectional outputs); shape=\ *(maxLength, batch_size,
#    hiddenSize)*
# -  ``hidden``: updated hidden state from GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hiddenSize)*
#
#

class EncoderRNN(nn.Module):

    def __init__(self, embeddingTokenSize, hiddenSize, embedding, noLayers=2, dropout=0.1):
        # print(input_size)
        super(EncoderRNN, self).__init__()
        self.n_layers = noLayers
        self.hiddenSize = hiddenSize
        self.embeddingTokenSize = embeddingTokenSize
        self.embedding = embedding

        self.gru = nn.GRU(hiddenSize, hiddenSize, noLayers,
                          dropout=(0 if noLayers == 1 else dropout), bidirectional=True)

        if embedding == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hiddenSize)
        elif embedding == embeddingType.glove:
            self.embeddingLayer = GloveEmbeddings(embeddingTokenSize)
        else:
            self.embeddingLayer = lambda x: x

    def forward(self, input, inputLengths, hidden=None):
        embedded = self.embeddingLayer(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, self.batchSize, self.hiddenSize, device=device)


######################################################################
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. A common problem with a vanilla seq2seq decoder is that
# if we rely solely on the context vector to encode the entire input
# sequence’s meaning, it is likely that we will have information loss.
# This is especially the case when dealing with long input sequences,
# greatly limiting the capability of our decoder.
#
# To combat this, `Bahdanau et al. <https://arxiv.org/abs/1409.0473>`__
# created an “attention mechanism” that allows the decoder to pay
# attention to certain parts of the input sequence, rather than using the
# entire fixed context at every step.
#
# At a high level, attention is calculated using the decoder’s current
# hidden state and the encoder’s outputs. The output attention weights
# have the same shape as the input sequence, allowing us to multiply them
# by the encoder outputs, giving us a weighted sum which indicates the
# parts of encoder output to pay attention to. `Sean
# Robertson’s <https://github.com/spro>`__ figure describes this very
# well:
#
# .. figure:: /_static/img/chatbot/attn2.png
#    :align: center
#    :alt: attn2
#
# `Luong et al. <https://arxiv.org/abs/1508.04025>`__ improved upon
# Bahdanau et al.’s groundwork by creating “Global attention”. The key
# difference is that with “Global attention”, we consider all of the
# encoder’s hidden states, as opposed to Bahdanau et al.’s “Local
# attention”, which only considers the encoder’s hidden state from the
# current time step. Another difference is that with “Global attention”,
# we calculate attention weights, or energies, using the hidden state of
# the decoder from the current time step only. Bahdanau et al.’s attention
# calculation requires knowledge of the decoder’s state from the previous
# time step. Also, Luong et al. provides various methods to calculate the
# attention energies between the encoder output and decoder output which
# are called “score functions”:
#
# .. figure:: /_static/img/chatbot/scores.png
#    :width: 60%
#    :align: center
#    :alt: scores
#
# where :math:`h_t` = current target decoder state and :math:`\bar{h}_s` =
# all encoder states.
#
# Overall, the Global attention mechanism can be summarized by the
# following figure. Note that we will implement the “Attention Layer” as a
# separate ``nn.Module`` called ``Attn``. The output of this module is a
# softmax normalized weights tensor of shape *(batch_size, 1,
# maxLength)*.
#
# .. figure:: /_static/img/chatbot/global_attn.png
#    :align: center
#    :width: 60%
#    :alt: global_attn
#

# Luong attention layer
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


######################################################################
# Now that we have defined our attention submodule, we can implement the
# actual decoder model. For the decoder, we will manually feed our batch
# one time step at a time. This means that our embedded word tensor and
# GRU output will both have shape *(1, batch_size, hiddenSize)*.
#
# **Computation Graph:**
#
#    1) Get embedding of current input word.
#    2) Forward through unidirectional GRU.
#    3) Calculate attention weights from the current GRU output from (2).
#    4) Multiply attention weights to encoder outputs to get new "weighted sum" context vector.
#    5) Concatenate weighted context vector and GRU output using Luong eq. 5.
#    6) Predict next word using Luong eq. 6 (without softmax).
#    7) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_step``: one time step (one word) of input sequence batch;
#    shape=\ *(1, batch_size)*
# -  ``last_hidden``: final hidden layer of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hiddenSize)*
# -  ``encoderOutputs``: encoder model’s output; shape=\ *(maxLength,
#    batch_size, hiddenSize)*
#
# **Outputs:**
#
# -  ``output``: softmax normalized tensor giving probabilities of each
#    word being the correct next word in the decoded sequence;
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: final hidden state of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hiddenSize)*
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hiddenSize, embeddingTokenSize, embeddingVersion, noLayers=2, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.hiddenSize = hiddenSize
        self.embeddingTokenSize = embeddingTokenSize
        self.embeddingVersion = embeddingVersion
        self.dropout = dropout
        self.maxLength = maxLenSentence
        self.noLayers = noLayers

        if embeddingVersion == embeddingType.indices:
            self.embeddingLayer = nn.Embedding(self.embeddingTokenSize, self.hiddenSize)
        elif embeddingVersion == embeddingType.glove:
            self.embeddingLayer = GloveEmbeddings(embeddingTokenSize)
        else:
            self.embeddingLayer = lambda x: x

        self.attention = AttentionModel(hiddenSize)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.gru = nn.GRU(hiddenSize, hiddenSize, self.noLayers, dropout=(0 if self.noLayers == 1 else dropout))
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.embeddingTokenSize)

    def forward(self, input, hidden, encoderOutputs):
        # This is ran one word at a time rather than for the entire batch
        embedding = self.embeddingLayer(input)
        embedding = self.dropoutLayer(embedding)
        gruOutput, hidden = self.gru(embedding, hidden)
        attentionWeights = self.attention(gruOutput, encoderOutputs)
        context = attentionWeights.bmm(encoderOutputs.transpose(0, 1))
        gruOutput = gruOutput.squeeze(0)
        context = context.squeeze(1)
        concatGruContext = torch.cat((gruOutput, context), 1)
        output = torch.tanh(self.concat(concatGruContext))
        output = F.softmax(self.out(output), dim=1)
        return output, hidden


######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacherForcingRatio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacherForcingRatio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# Image source: Goodfellow et al. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you can run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoderOptimizer, decoderOptimizer, batch_size, clip):
    # Zero gradients
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoderOutputs, encoderHidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoderInput = torch.LongTensor([[SOS for _ in range(batch_size)]])
    decoderInput = decoderInput.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoderHidden = encoderHidden[:decoder.noLayers]

    # Determine if we are using teacher forcing this iteration
    useTeacherForcing = True if random.random() < teacherForcingRatio else False

    # Forward batch of sequences through decoder one time step at a time
    if useTeacherForcing:
        for t in range(max_target_len):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )
            # Teacher forcing: next input is current target
            decoderInput = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoderOutput, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoderOutput.topk(1)
            decoderInput = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoderInput = decoderInput.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoderOutput, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoderOptimizer.step()
    decoderOptimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def trainIters(datasetBatches, encoder, decoder, encoderOptimizer, decoderOptimizer, embedding,
               n_iteration, batch_size, print_every, clip):
    # Load batches for each iteration
    batchFetcher = datasetBatches.trainDL

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")

    print(f"One run through all batches is {len(batchFetcher) / n_iteration * 100}% of the required iterations")

    for iteration, batch in enumerate(batchFetcher):
        # Extract fields from batch
        inputEmbeddings = batch["inputEmbeddings"]
        outputEmbeddings = batch["outputEmbeddings"]
        inputIndices = batch["inputIndices"]
        outputIndices = batch["outputIndices"]
        lengths = batch["lengths"]
        maxSimplifiedLength = batch["maxSimplifiedLength"]
        mask = batch["mask"]

        # Run a training iteration with batch
        loss = train(inputIndices, lengths, outputIndices, mask, maxSimplifiedLength, encoder,
                     decoder, embedding, encoderOptimizer, decoderOptimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            printLossAvg = print_loss / print_every
            print(
                f"Iteration: {iteration}; Percent complete: {iteration / len(batchFetcher) * 100}%; Percent complete of n_iterations: {iteration / n_iteration * 100}%; Average loss: {printLossAvg}")
            print_loss = 0


######################################################################
# Run Model
# ---------
#
# Finally, it is time to run our model!
#
# Regardless of whether we want to train or test the chatbot model, we
# must initialize the individual encoder and decoder models. In the
# following block, we set our desired configurations, choose to start from
# scratch or set a checkpoint to load from, and build and initialize the
# models. Feel free to play with different model configurations to
# optimize performance.
#

# Configure models
encoder_n_layers = 2  #
decoder_n_layers = 2  #
dropout = 0.1  #
batch_size = 64  #

print('Building encoder and decoder ...')
# Initialize word embeddings
# Initialize encoder & decoder models
# Use appropriate device
encoder = EncoderRNN(embeddingTokenSize, hiddenSize, embeddingType.indices, noLayers=encoder_n_layers,
                     dropout=dropout).to(device)  #
decoder = AttnDecoderRNN(hiddenSize, embeddingTokenSize, embeddingType.indices, noLayers=decoder_n_layers,
                         dropout=dropout).to(device)  #
print('Models built and ready to go!')

######################################################################
# Run Training
# ~~~~~~~~~~~~
#
# Run the following block if you want to train the model.
#
# First we set training parameters, then we initialize our optimizers, and
# finally we call the ``trainIters`` function to run our training
# iterations.
#

# Configure training/optimization
clip = 50.0  #
teacherForcingRatio = 1.0  #
learning_rate = 0.0001  #
decoder_learning_ratio = 5.0  #
n_iteration = 10000  #
print_every = 1  #

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoderOptimizer = optim.Adam(encoder.parameters(), lr=learning_rate)  #
decoderOptimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)  #

# If you have cuda, configure cuda to call
for state in encoderOptimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoderOptimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(datasetBatches, encoder, decoder, encoderOptimizer, decoderOptimizer,
           embedding, n_iteration, batch_size, print_every, clip)
######################################################################
# Conclusion
# ----------
#
# That’s all for this one, folks. Congratulations, you now know the
# fundamentals to building a generative chatbot model! If you’re
# interested, you can try tailoring the chatbot’s behavior by tweaking the
# model and training parameters and customizing the data that you train
# the model on.
#
# Check out the other tutorials for more cool deep learning applications
# in PyTorch!
#
