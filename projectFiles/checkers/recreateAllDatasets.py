from projectFiles.preprocessing.loadDatasets.loadAsset import loadAsset
from projectFiles.preprocessing.loadDatasets.loadNewsela import loadNewsela
from projectFiles.preprocessing.loadDatasets.loadWikiLarge import loadWikiLarge
from projectFiles.preprocessing.loadDatasets.loadWikiSmall import loadWikiSmall

# This creates new Pickle files to be loaded up from - saves us creating all new classes from 1000s of txt files each time we run

loadWikiSmall(False, True)
loadWikiLarge(False, True)
loadAsset(False, True)
loadNewsela(False, True)
