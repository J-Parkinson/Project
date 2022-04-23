from torch.utils.data import DataLoader


# Stores training, dev and test set splits
class simplificationDatasetLoader():
    def __init__(self, simpDS, batch_size=256):
        self.dataset = simpDS.dataset
        self.trainDL = DataLoader(simpDS.train, batch_size=batch_size)
        self.devDL = DataLoader(simpDS.dev, batch_size=batch_size)
        self.testDL = DataLoader(simpDS.test, batch_size=batch_size)
