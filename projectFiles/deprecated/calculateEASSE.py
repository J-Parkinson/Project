from time import time

from matplotlib import ticker

from easse.quality_estimation import corpus_quality_estimation
from easse.samsa import corpus_samsa
from projectFiles.evaluation.easse.calculateEASSE import calculateFleschKincaid, calculateBLEU, \
    calculateAverageSentenceBLEU, calculateBERTScore, calculateSARI, calculateF1Token


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


def _sameSizeCreateSentenceLists(allOriginal, allSimplified, allPredicted):
    originalSmeared = [[sentence for _ in range(len(simplified))] for sentence, simplified in
                       zip(allOriginal, allSimplified)]
    predictedSmeared = [[sentence for _ in range(len(simplified))] for sentence, simplified in
                        zip(allPredicted, allSimplified)]

    simpFlat = [z for y in allSimplified for z in y]
    origSmeaFlat = [z for y in originalSmeared for z in y]
    predSmeaFlat = [z for y in predictedSmeared for z in y]

    return simpFlat, origSmeaFlat, predSmeaFlat


def computeReport(simplificationSets, testSet):
    for set in simplificationSets:
        set.original = " ".join(set.original)
        set.allSimple = [" ".join(sentence) for sentence in set.allSimple]

    allOriginal = [set.original for set in simplificationSets]
    allSimplifiedOriginal = [set.allSimple for set in simplificationSets]
    allSimplified = [list(token) for token in zip(*allSimplifiedOriginal)]
    allPredicted = [set.predicted for set in simplificationSets]

    write_html_report(f"report_{time.time()}.html", orig_sents=allOriginal, sys_sents=allPredicted,
                      refs_sents=allSimplified, test_set=testSet)
    return


def calculateSAMSA(originalSentences, predictedSentences):
    return corpus_samsa(orig_sents=originalSentences, sys_sents=predictedSentences)


def calculateOtherMetrics(originalSentences, systemSentences):
    return corpus_quality_estimation(orig_sentences=originalSentences, sys_sentences=systemSentences)


def staticValidation(allOriginal, allSimplifiedSets):
    allResults = {}
    allSimplifiedSetsSARI = [list(sent) for sent in zip(*allSimplifiedSets)]
    allSimplifiedFlat = [z for y in allSimplifiedSets for z in y]

    fkclOriginal = calculateFleschKincaid(allOriginal)
    allResults["Flesch-Kincaid scores for original sentences"] = fkclOriginal
    fkclSimplifiedExamples = calculateFleschKincaid(allSimplifiedFlat)
    allResults["Flesch-Kincaid scores for simplified example sentences"] = fkclSimplifiedExamples

    bleuScoreOriginal = calculateBLEU(allOriginal, allSimplifiedSets)
    allResults["BLEU score"] = bleuScoreOriginal
    bleuAverageOriginal = calculateAverageSentenceBLEU(allOriginal, allSimplifiedSets)
    allResults["Average BLEU score"] = bleuAverageOriginal

    bertScoreOriginal = calculateBERTScore(allOriginal, allSimplifiedSetsSARI)
    allResults["BERTscore `Precision/Recall/F1"] = bertScoreOriginal

    return allResults


def computeAll(allOriginal, allSimplifiedSets, allPredicted, samsa=False):
    allResults = {}
    allSimplifiedSetsSARI = [list(sent) for sent in zip(*allSimplifiedSets)]
    sameSizeSimplified, sameSizeOriginal, sameSizePredicted = \
        _sameSizeCreateSentenceLists(allOriginal, allSimplifiedSets, allPredicted)

    print("Computing FKCL")

    # Flesch-Kincaid metrics
    fkclOriginal = calculateFleschKincaid(allOriginal)
    allResults["Flesch-Kincaid scores for original sentences"] = fkclOriginal
    fkclSimplifiedExamples = calculateFleschKincaid(sameSizeSimplified)
    allResults["Flesch-Kincaid scores for simplified example sentences"] = fkclSimplifiedExamples
    fkclPredictions = calculateFleschKincaid(allPredicted)
    allResults["Flesch-Kincaid scores for predicted sentences"] = fkclPredictions

    print("Computing BLEU")
    # BLEU scores
    bleuScoreOriginal = calculateBLEU(allOriginal, allSimplified)
    allResults["BLEU score for original vs simplified sentences"] = bleuScoreOriginal
    bleuScorePredicted = calculateBLEU(allPredicted, allSimplified)
    allResults["BLEU score for predicted vs simplified sentences"] = bleuScorePredicted
    bleuAverageOriginal = calculateAverageSentenceBLEU(allOriginal, allSimplified)
    allResults["Average BLEU score for original vs simplified sentences"] = bleuAverageOriginal
    bleuAveragePredicted = calculateAverageSentenceBLEU(allPredicted, allSimplified)
    allResults["Average BLEU score for predicted vs simplified sentences"] = bleuAveragePredicted

    print("Computing SARI")
    # SARI scores
    sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel = calculateSARI(allOriginal, allPredicted, allSimplified,
                                                                         microSari=False)
    allResults["SARI macro avg/add/keep/delete scores"] = (sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel)
    sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel = calculateSARI(allOriginal, allPredicted, allSimplified,
                                                                         microSari=True)
    allResults["SARI micro avg/add/keep/delete scores"] = (sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel)

    print("Computing BERTscore")
    # BERTscore
    bertScoreOriginal = calculateBERTScore(allOriginal, allSimplified)
    allResults["BERTscore for original vs simplified sentences"] = bertScoreOriginal
    bertScorePredictions = calculateBERTScore(allPredicted, allSimplified)
    allResults["BERTscore for predicted vs simplified sentences"] = bertScorePredictions

    if samsa:
        print("Computing SAMSA")
        # SAMSA scores
        samsaOriginalSimplified = calculateSAMSA(sameSizeOriginal, sameSizeSimplified)
        allResults["SAMSA score for original vs simplified sentences"] = samsaOriginalSimplified
        samsaOriginalPredicted = calculateSAMSA(allOriginal, allPredicted)
        allResults["SAMSA score for original vs predicted sentences"] = samsaOriginalPredicted

    print("Computing token-wise F1")
    # Token-wise F1 score
    tokenF1Original = calculateF1Token(allOriginal, allSimplified)
    allResults["Token F1 score for original vs simplified sentences"] = tokenF1Original
    tokenF1Predicted = calculateF1Token(allPredicted, allSimplified)
    allResults["Token F1 score for predicted vs simplified sentences"] = tokenF1Predicted

    print("Computing other quality metrics")
    # Corpus quality metrics
    qualityOriginalSimp = {f"{k} for original vs simplified sentences": v for k, v in
                           calculateOtherMetrics(sameSizeOriginal, sameSizeSimplified).items()}
    allResults.update(qualityOriginalSimp)
    qualityOriginalPred = {f"{k} for original vs predicted sentences": v for k, v in
                           calculateOtherMetrics(allOriginal, allPredicted).items()}
    allResults.update(qualityOriginalPred)

    for resultName, result in allResults.items():
        print(f"{resultName}: {result}")
    return allResults
