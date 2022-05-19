from time import time

import pandas as pd
from matplotlib import pyplot as plt

from projectFiles.evaluation.loadExistingData import *


def allResultsToTables(allResults, caption=None):
    runs = list(allResults.keys())
    metrics = list(allResults[runs[0]].keys())
    metricHeaders = [metric.split("|")[0] for metric in metrics]
    print(metricHeaders)
    rawTable = [[allResults[run][metric] for metric in metrics] for run in runs]
    dataFrame = pd.DataFrame(rawTable, index=runs, columns=[metric.split("|")[0] for metric in metrics])
    print(dataFrame.to_latex(caption=caption))


def showPlotMultipleLines(xyPairs, labels, yAxisLabel, title, saveLoc, noBatchesInEpoch=None, noEpochs=None):
    ax1 = plt.subplot(1, 1, 1)
    plt.title(title)
    # this locator puts ticks at regular intervals
    ax1.plot()
    for i, xy in enumerate(xyPairs):
        xy = xy.items()
        print(xy)
        ax1.plot([k[0][0] for k in xy], [v[1] for v in xy], label=labels[i])
    ax1.set_ylabel(yAxisLabel)
    ax1.set_xlabel("Batch number")

    # Set scond x-axis
    if noBatchesInEpoch and noEpochs:
        ax2 = ax1.twiny()

        # Decide the ticklabel position in the new x-axis,
        # then convert them to the position in the old x-axis
        newlabel = list(range(noEpochs + 1))  # labels of the xticklabels: the position in the new x-axis
        k2degc = lambda x: x * noBatchesInEpoch  # convert function: from Kelvin to Degree Celsius
        newpos = [k2degc(x) for x in newlabel]  # position of the xticklabels in the old x-axis
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel)

        ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
        ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
        ax2.spines['bottom'].set_position(('outward', 36))
        ax2.set_xlabel('Epoch number')
        ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()
    ax1.legend()

    # Save the figure
    plt.savefig(f"{saveLoc}/{title}_{time()}.png", dpi=300)
    plt.show()


def allResultsLinesToGraph(allResultsGraphs, saveLoc, noBatchesInEpoch=None, noEpochs=None, labels=None,
                           yAxisLabel=None, title=None):
    runs = list(allResultsGraphs.keys())
    metrics = list(allResultsGraphs[runs[0]].keys())
    # Prepare data
    for metric in metrics:
        data = [allResultsGraphs[run][metric] for run in runs]
        labels = labels if labels else runs
        yAxisLabel = metric if metric else yAxisLabel
        title = metric if metric else title
        print(len(data))
        showPlotMultipleLines(data, labels, yAxisLabel, title, saveLoc, noBatchesInEpoch, noEpochs)


allResultsLinesToGraph(loadAllResultLines([
    ("WikiLarge / GloVe",
     r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_glove_140237"),
    ("WikiLarge / indices",
     r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_indices_075230"),
]), 4032, 4, r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project",
    labels=["GloVe", "Indices"])

# allResultsToTables(loadAllResultEpoch(
#    [
#        ("WikiLarge / GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_glove_140237\epoch4"),
#        ("WikiLarge / indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_indices_075230\epoch4"),
#        ("WikiSmall / GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiSmall_CL-randomized_glove_133015\epoch4"),
#        ("WikiSmall / indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiSmall_CL-randomized_indices_063322\epoch4"),
#        ("ASSET / GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\asset_CL-randomized_glove_131452\epoch4"),
#        ("ASSET / indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\asset_CL-randomized_indices_060449\epoch4")
#    ]
# ))
