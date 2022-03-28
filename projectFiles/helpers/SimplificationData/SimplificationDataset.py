from torch.utils.data import Dataset


class simplificationDataset(Dataset):
    def _sortBySimplificationAmount(self):
        return

    def __init__(self, simplificationPairSet, sortBySimplificationAmount=False):
        self.dataset = simplificationPairSet
        if sortBySimplificationAmount:
            self._sortBySimplificationAmount()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].getData()