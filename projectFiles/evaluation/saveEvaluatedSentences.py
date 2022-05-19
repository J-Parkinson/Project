import csv


def saveTestData(allData, results, fileSaveDir):
    with open(f"{fileSaveDir}/easse.txt", "w+") as file:
        for k, v in results.items():
            file.write(f"{k}:{v}\n")

    with open(f"{fileSaveDir}/evaluatedSentences.csv", 'w', encoding="utf-8") as file:
        spamWriter = csv.writer(file)
        spamWriter.writerow(
            ["Original", "Prediction", "Simplified sentences"])
        spamWriter.writerows([[line["input"], line["predicted"]] + line["outputs"] for line in allData])
