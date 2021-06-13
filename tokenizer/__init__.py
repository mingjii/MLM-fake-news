from tokenizer.bert_base_ch import pretrained_Tknzr
from tokenizer.char import Tknzr_char

TKNZR_OPT = {
    Tknzr_char.tknzr_name: Tknzr_char,
    pretrained_Tknzr.tknzr_name: pretrained_Tknzr
}
