AA_MASSES_DICT = {
    "<pad>": 0.0,
    "<mask" + ">": 0.0,
    "<mask_token>": 0.0,
    "A": 71.0371138,   # Alanine
    "R": 156.1011110,  # Arginine
    "N": 114.0429274,  # Asparagine
    "D": 115.0269431,  # Aspartic acid
    "C": 103.0091845,  # Cysteine
    "E": 129.0425931,  # Glutamic acid
    "Q": 128.0585775,  # Glutamine
    "G": 57.0214637,   # Glycine
    "H": 137.0589119,  # Histidine
    "I": 113.0840640,  # Isoleucine
    "L": 113.0840640,  # Leucine
    "K": 128.0949630,  # Lysine
    "M": 131.0404846,  # Methionine
    "F": 147.0684139,  # Phenylalanine
    "P": 97.0527639,   # Proline
    "S": 87.0320284,   # Serine
    "T": 101.0476785,  # Threonine
    "W": 186.0793130,  # Tryptophan
    "Y": 163.0633286,  # Tyrosine
    "V": 99.0684139,   # Valine
}

# Standard 20 amino acids in a stable order for vocabulary construction.
STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Monoisotopic mass of a proton (Da).
M_H = 1.007276

# Monoisotopic mass of water (Da).
M_H2O = 18.010565
