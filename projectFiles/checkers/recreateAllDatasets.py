from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsela import loadNewsela
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall

loadWikiSmall(False, True)
loadWikiLarge(False, True)
loadAsset(False, True)
loadNewsela(False, True)
