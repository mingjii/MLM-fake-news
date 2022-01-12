from tokenizer.char import Tknzr_char
from tokenizer.sentence_piece_hg import Tknzr_sentPiece
# from tokenizer.sentence_piece import Tknzr_sentPiece

TKNZR_OPT = {
    Tknzr_char.tknzr_name: Tknzr_char,
    Tknzr_sentPiece.tknzr_name: Tknzr_sentPiece,
}
