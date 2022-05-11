from time import time

from matplotlib import pyplot as plt, ticker

from projectFiles.seq2seq.evaluate import evaluate


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(f"attention_{time()}.png", dpi=fig.dpi)
    fig.show()


def evaluateAndShowAttention(encoder, decoder, inputSentence):
    output_words, attentions = evaluate(
        encoder, decoder, inputSentence)
    print('input =', inputSentence)
    print('output =', ' '.join(output_words))
    showAttention(inputSentence, output_words, attentions)

def showPlot(x, y, title, saveLoc):
    title = title.split("`")
    titleName = title[0]
    if len(title) == 1:
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(x, y)
        plt.title(titleName)
        plt.xlabel("Sentence number")
        plt.ylabel("Average loss per sentence")
        plt.savefig(f"{saveLoc}/loss_{time()}.png", dpi=fig.dpi)
        plt.show()
    else:
        titleLegend = title[1].split("/")
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        for i in range(len(titleLegend)):
            plt.plot(x, [y1[i] for y1 in y], label=titleLegend[i])
        plt.title(titleName)
        plt.xlabel("Sentence number")
        plt.ylabel("Average loss per sentence")
        plt.legend()
        plt.savefig(f"{saveLoc}/loss_{time()}.png", dpi=fig.dpi)
        plt.show()
