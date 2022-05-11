from projectFiles.helpers.DatasetToLoad import datasetToLoad
from projectFiles.helpers.curriculumLearningFlag import curriculumLearningFlag, curriculumLearningMetadata
from projectFiles.helpers.embeddingType import embeddingType
from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsela import loadNewsela
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall
from projectFiles.seq2seq.constants import maxLengthSentence
from projectFiles.seq2seq.initialiseCurriculumLearning import initialiseCurriculumLearning


def simplificationDataToPyTorch(dataset, embedding=embeddingType.indices, maxLen=maxLengthSentence):
    if dataset == datasetToLoad.asset:
        print("Loading ASSET")
        datasetLoaded = loadAsset()
    elif dataset == datasetToLoad.newsela:
        print("Loading Newsela")
        datasetLoaded = loadNewsela()
    elif dataset == datasetToLoad.wikilarge:
        print("Loading WikiLarge")
        datasetLoaded = loadWikiLarge()
    else:
        print("Loading WikiSmall")
        datasetLoaded = loadWikiSmall()

    print("Processing dataset")

    initialiseCurriculumLearning(datasetLoaded.train, curriculumLearningMetadata(curriculumLearningFlag.ordered))
    initialiseCurriculumLearning(datasetLoaded.dev, curriculumLearningMetadata(curriculumLearningFlag.ordered))
    initialiseCurriculumLearning(datasetLoaded.test, curriculumLearningMetadata(curriculumLearningFlag.ordered))

    # We need to convert the imported simplificationSets into either NLTK or BERT sets depending on the embedding
    datasetLoaded.loadFromPickleAndPadAndDeleteLong(embedding, maxLen)

    return datasetLoaded

#simplificationDataToPyTorch(datasetToLoad.asset)
