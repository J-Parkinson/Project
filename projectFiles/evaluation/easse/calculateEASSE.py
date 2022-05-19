from easse.bleu import corpus_bleu, corpus_averaged_sentence_bleu
from easse.compression import corpus_f1_token
from easse.fkgl import corpus_fkgl
from easse.sari import get_corpus_sari_operation_scores

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

def calculateF1Token(originalSentences, referenceSentenceSets):
    return corpus_f1_token(sys_sents=originalSentences, refs_sents=referenceSentenceSets)

#Calculates metrics for the validation set, stored in epochData
def computeValidationEvaluation(allOriginal, allSimplifiedSets, allPredicted):
    allResults = {}
    allSimplifiedSetsRefSent = [list(sent) for sent in zip(*allSimplifiedSets)]

    # Flesch-Kincaid metrics
    fkclPredictions = calculateFleschKincaid(allPredicted)
    allResults["FK score|Flesch-Kincaid scores for predicted sentences"] = fkclPredictions
    print(f"Flesch-Kincaid for predicted: {fkclPredictions}")

    # BLEU scores
    bleuScorePredicted = calculateBLEU(allPredicted, allSimplifiedSetsRefSent)
    allResults["BLEU score|BLEU score for predicted vs simplified sentences"] = bleuScorePredicted
    print(f"BLEU score: {bleuScorePredicted}")
    bleuAveragePredicted = calculateAverageSentenceBLEU(allPredicted, allSimplifiedSets)
    allResults["BLEU score|Average BLEU score for predicted vs simplified sentences"] = bleuAveragePredicted
    print(f"BLEU average score: {bleuAveragePredicted}")

    # SARI scores
    sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel = calculateSARI(allOriginal, allPredicted,
                                                                         allSimplifiedSetsRefSent, microSari=False)
    allResults["SARI macro score|SARI macro scores`Average/Add/Keep/Delete"] = (
    sariMacro, sariMacroAdd, sariMacroKeep, sariMacroDel)
    print(f"SARI macro scores average: {sariMacro}")
    sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel = calculateSARI(allOriginal, allPredicted,
                                                                         allSimplifiedSetsRefSent, microSari=True)
    allResults["SARI micro score|SARI micro scores`Average/Add/Keep/Delete"] = (
    sariMicro, sariMicroAdd, sariMicroKeep, sariMicroDel)
    print(f"SARI micro scores average: {sariMicro}")

    print("Computing token-wise F1")
    # Token-wise F1 score
    tokenF1Predicted = calculateF1Token(allPredicted, allSimplifiedSetsRefSent)
    allResults["F1 score|Token-wise F1 score for predicted vs simplified sentences"] = tokenF1Predicted

    return allResults
