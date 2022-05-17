import csv


def saveTestData(fileSaveDir, allData):
    with open(f"{fileSaveDir}/evaluatedSentences.csv", 'w', encoding="utf-8") as file:
        spamWriter = csv.writer(file)
        spamWriter.writerow(
            ["Original", "Prediction"] + [f"Simplified {n}" for n in range(len(allData[0]["output"]))])
        spamWriter.writerows([[line["input"], line["predicted"]] + line["output"] for line in allData])
