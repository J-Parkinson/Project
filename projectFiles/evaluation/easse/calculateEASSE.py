from easse.bertscore import corpus_bertscore
from easse.bleu import corpus_bleu, corpus_averaged_sentence_bleu
from easse.compression import corpus_f1_token
from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.samsa import corpus_samsa
from easse.sari import get_corpus_sari_operation_scores

# from easse.report import write_html_report
from projectFiles.seq2seq.evaluate import loadEncoderDecoderDatasetAndEvaluateAll


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

def computeAll(simplificationSets, samsa=False):
    allResults = {}

    for set in simplificationSets:
        set.original = " ".join(set.original)
        set.allSimple = [" ".join(sentence) for sentence in set.allSimple]

    allOriginal = [set.original for set in simplificationSets]
    allSimplifiedOriginal = [set.allSimple for set in simplificationSets]
    allSimplified = [list(token) for token in zip(*allSimplifiedOriginal)]
    allPredicted = [set.predicted for set in simplificationSets]
    sameSizeSimplified, sameSizeOriginal, sameSizePredicted = _sameSizeCreateSentenceLists(simplificationSets)

    print(f"Predicted run time: {int((len(allOriginal) + len(sameSizeSimplified)) / 60)} minutes")

    print("Computing FKCL")

    # Flesch-Kincaid metrics
    fkclOriginal = corpus_fkgl(sentences=allOriginal)
    allResults["Flesch-Kincaid scores for original sentences"] = fkclOriginal
    fkclSimplifiedExamples = corpus_fkgl(sentences=sameSizeSimplified)
    allResults["Flesch-Kincaid scores for simplified example sentences"] = fkclSimplifiedExamples
    fkclPredictions = corpus_fkgl(sentences=allPredicted)
    allResults["Flesch-Kincaid scores for predicted sentences"] = fkclPredictions

    print("Computing BLEU")
    # BLEU scores
    bleuScoreOriginal = corpus_bleu(sys_sents=allOriginal,
                                    refs_sents=allSimplified)
    allResults["BLEU score for original vs simplified sentences"] = bleuScoreOriginal
    bleuScorePredicted = corpus_bleu(sys_sents=allPredicted,
                                     refs_sents=allSimplified)
    allResults["BLEU score for predicted vs simplified sentences"] = bleuScorePredicted
    bleuAverageOriginal = corpus_averaged_sentence_bleu(sys_sents=allOriginal,
                                                        refs_sents=allSimplified)
    allResults["Average BLEU score for original vs simplified sentences"] = bleuAverageOriginal
    bleuAveragePredicted = corpus_averaged_sentence_bleu(sys_sents=allPredicted,
                                                         refs_sents=allSimplified)
    allResults["Average BLEU score for predicted vs simplified sentences"] = bleuAveragePredicted

    print("Computing SARI")
    # SARI scores
    sariMacroAdd, sariMacroKeep, sariMacroDel = get_corpus_sari_operation_scores(orig_sents=allOriginal,
                                                                                 sys_sents=allPredicted,
                                                                                 refs_sents=allSimplified,
                                                                                 use_paper_version=False)
    sariMacro = (sariMacroAdd + sariMacroKeep + sariMacroDel) / 3
    allResults["SARI macro add/keep/delete/avg scores"] = (sariMacroAdd, sariMacroKeep, sariMacroDel, sariMacro)
    sariMicroAdd, sariMicroKeep, sariMicroDel = get_corpus_sari_operation_scores(orig_sents=allOriginal,
                                                                                 sys_sents=allPredicted,
                                                                                 refs_sents=allSimplified,
                                                                                 use_paper_version=True)
    sariMicro = (sariMicroAdd + sariMicroKeep + sariMicroDel) / 3
    allResults["SARI micro add/keep/delete/avg scores"] = (sariMicroAdd, sariMicroKeep, sariMicroDel, sariMicro)

    print("Computing BERTscore")
    # BERTscore
    bertScoreOriginal = corpus_bertscore(sys_sents=allOriginal, refs_sents=allSimplified)
    allResults["BERTscore for original vs simplified sentences"] = bertScoreOriginal
    bertScorePredictions = corpus_bertscore(sys_sents=allPredicted, refs_sents=allSimplified)
    allResults["BERTscore for predicted vs simplified sentences"] = bertScorePredictions

    if samsa:
        print("Computing SAMSA")
        # SAMSA scores
        samsaOriginalSimplified = corpus_samsa(orig_sents=sameSizeOriginal, sys_sents=sameSizeSimplified)
        allResults["SAMSA score for original vs simplified sentences"] = samsaOriginalSimplified
        samsaOriginalPredicted = corpus_samsa(orig_sents=allOriginal, sys_sents=allPredicted)
        allResults["SAMSA score for original vs predicted sentences"] = samsaOriginalPredicted

    print("Computing token-wise F1")
    # Token-wise F1 score
    tokenF1Original = corpus_f1_token(sys_sents=allOriginal, refs_sents=allSimplified)
    allResults["Token F1 score for original vs simplified sentences"] = tokenF1Original
    tokenF1Predicted = corpus_f1_token(sys_sents=allPredicted, refs_sents=allSimplified)
    allResults["Token F1 score for predicted vs simplified sentences"] = tokenF1Predicted

    print("Computing other quality metrics")
    # Corpus quality metrics
    qualityOriginalSimp = {f"{k} for original vs simplified sentences": v for k, v in
                           corpus_quality_estimation(orig_sentences=sameSizeOriginal,
                                                     sys_sentences=sameSizeSimplified).items()}
    allResults.update(qualityOriginalSimp)
    qualityOriginalPred = {f"{k} for original vs predicted sentences": v for k, v in
                           corpus_quality_estimation(orig_sentences=allOriginal, sys_sentences=allPredicted).items()}
    allResults.update(qualityOriginalPred)

    for resultName, result in allResults.items():
        print(f"{resultName}: {result}")
    return allResults


evaluatedModelData = loadEncoderDecoderDatasetAndEvaluateAll("seq2seq/trainedModels/optimal_asset_025043")
computeAll(evaluatedModelData)
