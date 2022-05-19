metrics = [("BLEU score for predicted vs simplified sentences", "BLEU"),
           ("Flesch-Kincaid scores for predicted sentences", "FK score"),
           ("plotData", "Training loss"),
           ("plotDataDev", "Dev loss"),
           ("Add SARI macro scores", "SARI macro add"),
           ("Average SARI macro scores", "SARI macro average"),
           ("Delete SARI macro scores", "SARI macro delete"),
           ("Keep SARI macro scores", "SARI macro keep"),
           ("Add SARI micro scores", "SARI micro add"),
           ("Average SARI micro scores", "SARI micro average"),
           ("Delete SARI micro scores", "SARI micro delete"),
           ("Keep SARI micro scores", "SARI micro keep"),
           ("Token-wise F1 score for predicted vs simplified sentences", "Token F1")]

metricsLines = [("BLEU score for predicted vs simplified sentences", "BLEU"),
                ("Flesch-Kincaid scores for predicted sentences", "FK score"),
                ("plotData", "Training loss"),
                ("plotDataDev", "Dev loss"),
                ("SARI macro scores_Add", "SARI macro add"),
                ("SARI macro scores_Average", "SARI macro average"),
                ("SARI macro scores_Delete", "SARI macro delete"),
                ("SARI macro scores_Keep", "SARI macro keep"),
                ("SARI micro scores_Add", "SARI micro add"),
                ("SARI micro scores_Average", "SARI micro average"),
                ("SARI micro scores_Delete", "SARI micro delete"),
                ("SARI micro scores_Keep", "SARI micro keep"),
                ("Token-wise F1 score for predicted vs simplified sentences", "Token F1")]


def loadResultLines(fileLocation):
    results = {}
    for metric, key in metricsLines:
        with open(f"{fileLocation}/{metric}.txt", "r+") as metricData:
            metricDataRead = metricData.readlines()
            rawData = {eval(met.split(") ")[0] + ")"): eval(met.split(") ")[1][:-1]) for met in metricDataRead}
            results[key] = rawData
    return results


def loadResultEpoch(fileLocation):
    with open(f"{fileLocation}/easse.txt", "r+") as epochData:
        epochDataRead = epochData.readlines()
        data = {val.split(":")[0]: eval(val.split(":")[1][:-1]) for val in epochDataRead}
        dataSplit = {}
        for k, v in data.items():
            if len(k.split("`")) == 1:
                found = False
                for metric in metrics:
                    kNew = k.replace(metric[0], metric[1])
                    if kNew != k:
                        found = True
                    k = kNew
                if found:
                    dataSplit[k] = v
            else:
                pep = k.split("`")[1].split("/")
                for i, p in enumerate(pep):
                    key = f"{k.split('`')[0].split('|')[0]}|{p} {k.split('`')[0].split('|')[1]}"
                    for metric in metrics:
                        kNew = key.replace(metric[0], metric[1])
                        if kNew != key:
                            found = True
                        key = kNew
                    if found:
                        dataSplit[key] = v[i]
    return dataSplit


def loadAllResultEpoch(MN):
    return {name: loadResultEpoch(location) for name, location in MN}


def loadAllResultLines(MN):
    return {name: loadResultLines(location) for name, location in MN}

# print(loadResultLines(r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_glove_140237"))
# print(loadResultEpoch(r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_glove_140237\epoch4"))
# loadAllResultEpoch(
#    [
#        ("WikiLarge with GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_glove_140237\epoch4"),
#        ("WikiLarge with indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiLarge_CL-randomized_indices_075230\epoch4"),
#        ("WikiSmall with GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiSmall_CL-randomized_glove_133015\epoch4"),
#        ("WikiSmall with indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\wikiSmall_CL-randomized_indices_063322\epoch4"),
#        ("ASSET with GloVe", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\asset_CL-randomized_glove_131452\epoch4"),
#        ("ASSET with indices", r"C:\Users\jrp32\Documents\Cambridge University\Year III\Project\projectFiles\seq2seq\trainedModels\baselines\asset_CL-randomized_indices_060449\epoch4")
#    ]
# )
