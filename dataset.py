from torch.utils.data import Dataset
from datasets import load_dataset

class SSTDataset(Dataset):

    def __init__(self, split):
        assert split in ('train', 'validation', 'test')
        self.data = load_dataset('sst2',split=split)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate(batch):
        return [item for item in batch]