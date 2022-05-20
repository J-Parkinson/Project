import os
from csv import reader

from bert_score import BERTScorer

from projectFiles.constants import projectLoc

scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def addBertScoreToEasse(location):
    with open(f"{location}/evaluatedSentences.csv", "r+") as evaluatedSentences:
        readSentences = reader(evaluatedSentences)
        readData = [row for row in readSentences if row != []][1:]
        prediction = [row[1] for row in readData]
        simplified = [row[2:] for row in readData]
    P, R, F1 = scorer.score(prediction, simplified)
    corpusP, corpusR, corpusF1 = P.mean(), R.mean(), F1.mean()
    corpusP, corpusR, corpusF1 = (corpusP + 1) / 2, (corpusR + 1) / 2, (corpusF1 + 1) / 2
    with open(f"{location}/easse.txt", "a+") as easse:
        easse.write(
            f"BERTscore|BERTscore for predicted vs simplified sentences`Precision/Recall/F1:({corpusP},{corpusR},{corpusF1})\n")


def runBertScoreForEveryEpochInLoc(location):
    for epoch in list(os.walk(location))[0][1]:
        addBertScoreToEasse(f"{location}/{epoch}")


def runBertScoreForEveryModelInLoc(location):
    for model in list(os.walk(location))[0][1]:
        runBertScoreForEveryEpochInLoc(f"{location}/{model}")


def modelToTxt(location, batchesInEpoch):
    p = {}
    r = {}
    f1 = {}
    for epoch in list(os.walk(location))[0][1]:
        file = f"{location}/{epoch}/easse.txt"
        epochNo = int(epoch.split("epoch")[1])
        with open(file, "r+") as data:
            data = data.read().split("\n")[-2]
        p[((epochNo - 1) * batchesInEpoch + 1, 1, 1)] = data.split("(")[1].split(",")[0]
        r[((epochNo - 1) * batchesInEpoch + 1, 1, 1)] = data.split("(")[1].split(",")[1]
        f1[((epochNo - 1) * batchesInEpoch + 1, 1, 1)] = data.split("(")[1].split(",")[2].strip(")")
    with open(f"{location}/BERTscore precision.txt", "w+") as precision:
        precision.write("\n".join(f"{k} {v}" for k, v in p.items()))
    with open(f"{location}/BERTscore recall.txt", "w+") as precision:
        precision.write("\n".join(f"{k} {v}" for k, v in r.items()))
    with open(f"{location}/BERTscore F1.txt", "w+") as precision:
        precision.write("\n".join(f"{k} {v}" for k, v in f1.items()))
    return p, r, f1


# runBertScoreForEveryModelInLoc(f"{projectLoc}/seq2seq/trainedModels/baselines")
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/asset_CL-randomized_glove_131452", 283)
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/asset_CL-randomized_indices_060449", 283)
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/wikiLarge_CL-randomized_glove_140237", 3566)
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/wikiLarge_CL-randomized_indices_075230", 3566)
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/wikiSmall_CL-randomized_glove_133015", 903)
modelToTxt(f"{projectLoc}/seq2seq/trainedModels/baselines/wikiSmall_CL-randomized_indices_063322", 903)
