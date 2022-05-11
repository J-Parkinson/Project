from easse.bertscore import corpus_bertscore
from easse.bleu import corpus_bleu, corpus_averaged_sentence_bleu
from easse.compression import corpus_f1_token
from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.samsa import corpus_samsa
from easse.sari import get_corpus_sari_operation_scores

# Wrapper functions which handle evaluation using EASSE (and abstracts away complex optional parameters)

# This duplicates original and predicted sentences to pair up with simplified sentences
def _sameSizeCreateSentenceLists(simplificationSets):
    allOriginal = [set.original for set in simplificationSets]
    allSimplified = [set.allSimple for set in simplificationSets]
    allPredicted = [set.predicted for set in simplificationSets]

    originalSmeared = [[sentence for _ in range(len(simplified))] for sentence, simplified in
                       zip(allOriginal, allSimplified)]
    predictedSmeared = [[sentence for _ in range(len(simplified))] for sentence, simplified in
                        zip(allPredicted, allSimplified)]

    simpFlat = [z for y in allSimplified for z in y]
    origSmeaFlat = [z for y in originalSmeared for z in y]
    predSmeaFlat = [z for y in predictedSmeared for z in y]

    return simpFlat, origSmeaFlat, predSmeaFlat


# def computeReport(simplificationSets, testSet):
#    for set in simplificationSets:
#        set.original = " ".join(set.original)
#        set.allSimple = [" ".join(sentence) for sentence in set.allSimple]

#    allOriginal = [set.original for set in simplificationSets]
#    allSimplifiedOriginal = [set.allSimple for set in simplificationSets]
#    allSimplified = [list(token) for token in zip(*allSimplifiedOriginal)]
#    allPredicted = [set.predicted for set in simplificationSets]

#    write_html_report(f"report_{time.time()}.html", orig_sents=allOriginal, sys_sents=allPredicted, refs_sents=allSimplified, test_set=testSet)
#    return

def calculateFleschKincaid(sentences):
    return corpus_fkgl(sentences=sentences)


def calculateBLEU(inputSentences, referenceSentenceSets):
    return corpus_bleu(sys_sents=inputSentences, refs_sents=referenceSentenceSets)


def calculateAverageSentenceBLEU(inputSentences, referenceSentenceSets):
    return corpus_averaged_sentence_bleu(sys_sents=inputSentences, refs_sents=referenceSentenceSets)


def calculateSARI(originalSentences, predictedSentences, referenceSentenceSets, microSari=False):
    sariMacroAdd, sariMacroKeep, sariMacroDel = get_corpus_sari_operation_scores(orig_sents=originalSentences,
                                                                                 sys_sents=predictedSentences,
                                                                                 refs_sents=referenceSentenceSets,
                                                                                 use_paper_version=microSari)
    sariMacro = (sariMacroAdd + sariMacroKeep + sariMacroDel) / 3
    return sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel


def calculateBERTScore(originalSentences, referenceSentenceSets):
    return corpus_bertscore(sys_sents=originalSentences, refs_sents=referenceSentenceSets)


def calculateSAMSA(originalSentences, predictedSentences):
    return corpus_samsa(orig_sents=originalSentences, sys_sents=predictedSentences)


def calculateF1Token(originalSentences, referenceSentenceSets):
    return corpus_f1_token(sys_sents=originalSentences, refs_sents=referenceSentenceSets)


def calculateOtherMetrics(originalSentences, systemSentences):
    return corpus_quality_estimation(orig_sentences=originalSentences, sys_sentences=systemSentences)


#Calculates metrics for the validation set, stored in epochData
def computeValidation(allOriginal, allSimplifiedSets, allPredicted):
    allResults = {}

    allSimplified = [list(token) for token in zip(*allSimplifiedSets)]
    allSimplifiedFlat = [z for y in allSimplifiedSets for z in y]

    print("Computing FKCL")

    # Flesch-Kincaid metrics
    fkclOriginal = calculateFleschKincaid(allOriginal)
    allResults["Flesch-Kincaid scores for original sentences"] = fkclOriginal
    fkclSimplifiedExamples = calculateFleschKincaid(allSimplifiedFlat)
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
    allResults["SARI macro scores`Average/Add/Keep/Delete"] = (sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel)
    sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel = calculateSARI(allOriginal, allPredicted, allSimplified,
                                                                         microSari=True)
    allResults["SARI micro scores`Average/Add/Keep/Delete"] = (sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel)

    print("Computing BERTscore")
    # BERTscore
    bertScoreOriginal = calculateBERTScore(allOriginal, allSimplified)
    allResults["BERTscore for original vs simplified sentences`Precision/Recall/F1"] = bertScoreOriginal
    bertScorePredictions = calculateBERTScore(allPredicted, allSimplified)
    allResults["BERTscore for predicted vs simplified sentences`Precision/Recall/F1"] = bertScorePredictions

    return allResults


#Evaluation function for test set
def computeAll(allOriginal, allSimplifiedOriginal, allPredicted, samsa=False):
    allResults = {}

    allSimplified = [list(token) for token in zip(*allSimplifiedOriginal)]
    sameSizeSimplified, sameSizeOriginal, sameSizePredicted = _sameSizeCreateSentenceLists(simplificationSets)

    constantTimePred = 1 if samsa else 0.05
    print(f"Predicted run time: {int((len(allOriginal) + len(sameSizeSimplified)) / 60 * constantTimePred)} minutes")

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

# evaluatedModelData = loadEncoderDecoderDatasetAndEvaluateAll("seq2seq/trainedModels/optimal_asset_025043")
# computeAll(evaluatedModelData)
