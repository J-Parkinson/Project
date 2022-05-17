from time import time

from matplotlib import pyplot as plt


def showPlot(x, y, title, noBatchesInEpoch, noEpochs, saveLoc):
    [yAxisLabel, title] = title.split("|")
    title = title.split("`")
    titleName = title[0]
    if len(title) == 1:
        # Plot the data
        ax1 = plt.subplot(1, 1, 1)
        plt.title(titleName)
        ax1.plot([xk[0] for xk in x], y)
        ax1.set_ylabel(yAxisLabel)
        ax1.set_xlabel("Batch number")

        # Set scond x-axis
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

        # Save the figure
        plt.show()
        plt.savefig(f"{saveLoc}/loss_{time()}.png", dpi=300)
    else:
        titleLegend = title[1].split("/")
        ax1 = plt.subplot(1, 1, 1)
        plt.title(titleName)
        # this locator puts ticks at regular intervals
        ax1.plot()
        for i in range(len(titleLegend)):
            ax1.plot([xk[0] for xk in x], [y1[i] for y1 in y], label=titleLegend[i])
        ax1.set_ylabel(yAxisLabel)
        ax1.set_xlabel("Batch number")

        # Set scond x-axis
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

        ax1.legend()
        plt.tight_layout()

        # Save the figure
        plt.show()
        plt.savefig(f"{saveLoc}/loss_{time()}.png", dpi=300)


def printPlots(results, plotLosses, plotDevLosses, datasetName, embedding, fileSaveDir, trainingBatchesSize, noEpochs):
    sizeTrainingBatch = trainingBatchesSize
    print("Making plots")
    plotLossesFormatted = {(k[0], (k[0] - 1) / sizeTrainingBatch): v for k, v in plotLosses.items()}
    plotDevLossesFormatted = {(k[0], (k[0] - 1) / sizeTrainingBatch): v for k, v in plotDevLosses.items()}
    showPlot(*list(zip(*plotLossesFormatted.items())),
             f"Average training loss|Training set losses for {datasetName} with {embedding.name} embeddings",
             trainingBatchesSize, noEpochs,
             fileSaveDir)
    showPlot(*list(zip(*plotDevLossesFormatted.items())),
             f"Average development loss|Development set losses for {datasetName} with {embedding.name} embeddings",
             trainingBatchesSize, noEpochs,
             fileSaveDir)

    if len(results) > 0:
        metrics = list(results[(1, 1, 1)].keys())
        for metric in metrics:
            data = {k: v[metric] for k, v in results.items()}
            xPos = list(data.keys())
            yPos = list(data.values())
            showPlot(xPos, yPos,
                     metric,
                     trainingBatchesSize, noEpochs,
                     fileSaveDir)

# printPlots({(1, 1, 1): {'FK score|Flesch-Kincaid scores for predicted sentences': 14.21690625828191, 'FK score|BLEU score for predicted vs simplified sentences': 5.273178342926995, 'FK score|Average BLEU score for predicted vs simplified sentences': 10.662747803549998, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (27.044690391910635, 0.02760905577029266, 20.142030817497574, 60.96443130246404), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (27.488330773457307, 0.02760905577029266, 20.655368793801664, 61.78201447079997), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.08952658623456955, -0.0013873130083084106, -0.06150003895163536), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 18.72500314301209}, (76, 76, 1): {'FK score|Flesch-Kincaid scores for predicted sentences': 11.319407033596192, 'FK score|BLEU score for predicted vs simplified sentences': 1.235356399341275, 'FK score|Average BLEU score for predicted vs simplified sentences': 3.057466118524469, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (26.452625693809967, 0.08305647840531562, 18.753336207610335, 60.52148439541426), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (26.80955297753992, 0.08305647840531562, 19.14190981040862, 61.203692643805816), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.726154625415802, -0.02154531329870224, -0.42367634177207947), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 15.27095331681973}, (151, 151, 1): {'FK score|Flesch-Kincaid scores for predicted sentences': 12.095188957495786, 'FK score|BLEU score for predicted vs simplified sentences': 1.3027931806678559, 'FK score|Average BLEU score for predicted vs simplified sentences': 3.7286276806351086, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (27.593596137884806, 1.940514729866123, 20.28790598589794, 60.55236769789036), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (28.00823524994354, 1.9475013107104273, 20.93743529576546, 61.13976914335472), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.5862067341804504, -0.01817444898188114, -0.3287256956100464), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 17.919483894738132}, (226, 226, 1): {'FK score|Flesch-Kincaid scores for predicted sentences': 5.352008585365855, 'FK score|BLEU score for predicted vs simplified sentences': 5.4567948185537345, 'FK score|Average BLEU score for predicted vs simplified sentences': 4.618725702513762, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (30.039840379457264, 2.267451944440509, 26.55018746369415, 61.30188173023714), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (30.68348871211506, 2.2712679815381, 27.94678562808532, 61.83241252672176), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.2732911705970764, 0.14714036881923676, -0.08211736381053925), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 24.17406728712215}, (232, 1, 2): {'FK score|Flesch-Kincaid scores for predicted sentences': 6.574496773615348, 'FK score|BLEU score for predicted vs simplified sentences': 4.693529120648382, 'FK score|Average BLEU score for predicted vs simplified sentences': 4.618725702513762, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (29.526859598483952, 2.466210630326423, 25.011720319812074, 61.10264784531336), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (30.198227547279544, 2.470857730824161, 26.486426214209224, 61.63739869680525), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.28223758935928345, 0.133259579539299, -0.09393742680549622), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 23.69199472896466}, (307, 76, 2): {'FK score|Flesch-Kincaid scores for predicted sentences': 4.596045049019974, 'FK score|BLEU score for predicted vs simplified sentences': 8.08673942800124, 'FK score|Average BLEU score for predicted vs simplified sentences': 15.966613526045181, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (31.391435508795457, 2.1073153673322578, 30.359011528022904, 61.70797963103121), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (31.98157981246952, 2.1115835152452322, 31.6248258136337, 62.20833010852963), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.09079689532518387, 0.16851893067359924, 0.028547927737236023), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 28.518776982603967}, (382, 151, 2): {'FK score|Flesch-Kincaid scores for predicted sentences': 4.612749437485309, 'FK score|BLEU score for predicted vs simplified sentences': 7.941849421333463, 'FK score|Average BLEU score for predicted vs simplified sentences': 20.423020310820867, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (31.002173138432536, 1.7550110002528714, 29.82115265916736, 61.43035575587737), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (31.628526829202702, 1.7653958421078753, 31.216897893943347, 61.90328675155688), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (-0.0766945332288742, 0.1625462919473648, 0.03303153067827225), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 29.49443870846548}, (457, 226, 2): {'FK score|Flesch-Kincaid scores for predicted sentences': 1.7090053285968025, 'FK score|BLEU score for predicted vs simplified sentences': 10.837278814038996, 'FK score|Average BLEU score for predicted vs simplified sentences': 16.854189111095234, 'FK score|SARI macro scores`Average/Add/Keep/Delete': (31.633999312822464, 1.2188450895244267, 32.0625074454449, 61.62064540349807), 'FK score|SARI micro scores`Average/Add/Keep/Delete': (32.23148653806235, 1.224557137964834, 33.39997259868399, 62.06992987753823), 'FK score|BERTscore for predicted vs simplified sentences`Precision/Recall/F1': (0.033460795879364014, 0.17606185376644135, 0.09585969150066376), 'FK score|Token F1 score for original vs simplified sentences': 89.94881095688515, 'FK score|Token F1 score for predicted vs simplified sentences': 34.5191682767724}},
#           {(1, 1, 1): 9.360475274625522, (76, 76, 1): 7.040530521179191, (151, 151, 1): 13.353191258852165, (226, 226, 1): 19.33900614422476, (232, 1, 2): 5.7323613701776175, (307, 76, 2): 5.723816983551737, (382, 151, 2): 11.131394079095974, (457, 226, 2): 16.30896535400524},
#           {(1, 1, 1): 111.98563121605561, (76, 76, 1): 86.65435356488891, (151, 151, 1): 98.60607354500954, (226, 226, 1): 99.37558600111765, (232, 1, 2): 99.61350303420863, (307, 76, 2): 103.5559262046617, (382, 151, 2): 106.05028030697244, (457, 226, 2): 104.29704600783977},
#           "asset",
#           embeddingType.indices,
#           f'{projectLoc}/seq2seq/trainedModels/testPlots',
#           231,
#           2)
