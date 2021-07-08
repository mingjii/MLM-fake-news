from dataset.news import Seq2SeqNewsDataset
from dataset.mlm import MLMDataset

DSET_OPT = {
    Seq2SeqNewsDataset.dset_name: Seq2SeqNewsDataset,
    MLMDataset.dset_name: MLMDataset
}
