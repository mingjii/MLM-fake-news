from dataset.news import NewsDataset
from dataset.mlm import MLMDataset

DSET_OPT = {
    NewsDataset.dset_name: NewsDataset,
    MLMDataset.dset_name: MLMDataset
}
