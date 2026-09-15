AA_MASSES_DICT = {
    "<pad>": 0.0,
    "<mask" + ">": 0.0,
    "<mask_token>": 0.0,
    "A": 71.0371138,   # Alanine
    "R": 156.1011110,  # Arginine
    "N": 114.0429274,  # Asparagine
    "D": 115.0269431,  # Aspartic acid
    "C": 160.0306482,  # Cysteine (carbamidomethylated, standard in ProteomeTools)
    "C(unmod)": 103.0091845,  # Unmodified Cysteine
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
    # --- Post-Translational Modifications (PTMs): Residue-Level ---
    # Methionine Oxidation (+15.9949 Da, UNIMOD:35)
    "M(ox)": 147.0353992,
    "M(+15.99)": 147.0353992,
    "M(+15.995)": 147.0353992,
    "M[UNIMOD:35]": 147.0353992,
    # Cysteine Carboxyamidomethylation (+57.0215 Da, UNIMOD:4)
    "C(cam)": 160.0306482,
    "C(+57.02)": 160.0306482,
    "C(+57.021)": 160.0306482,
    "C[UNIMOD:4]": 160.0306482,
    # Asparagine Deamidation (+0.9840 Da, UNIMOD:7)
    "N(deam)": 115.0269434,
    "N(+0.98)": 115.0269434,
    "N(+.98)": 115.0269434,
    "N[UNIMOD:7]": 115.0269434,
    # Glutamine Deamidation (+0.9840 Da, UNIMOD:7)
    "Q(deam)": 129.0425935,
    "Q(+0.98)": 129.0425935,
    "Q(+.98)": 129.0425935,
    "Q[UNIMOD:7]": 129.0425935,
    # Serine Phosphorylation (+79.9663 Da, UNIMOD:21)
    "S(ph)": 166.9983594,
    "S(+79.97)": 166.9983594,
    "S(p)": 166.9983594,
    "S[UNIMOD:21]": 166.9983594,
    # Threonine Phosphorylation (+79.9663 Da, UNIMOD:21)
    "T(ph)": 181.0140095,
    "T(+79.97)": 181.0140095,
    "T(p)": 181.0140095,
    "T[UNIMOD:21]": 181.0140095,
    # Tyrosine Phosphorylation (+79.9663 Da, UNIMOD:21)
    "Y(ph)": 243.0296596,
    "Y(+79.97)": 243.0296596,
    "Y(p)": 243.0296596,
    "Y[UNIMOD:21]": 243.0296596,
    # --- Post-Translational Modifications (PTMs): N-Terminal ---
    # N-terminal Acetylation (+42.0106 Da, UNIMOD:1)
    "[UNIMOD:1]": 42.010565,
    "(+42.01)": 42.010565,
    "[+42.01]": 42.010565,
    "(+42.0106)": 42.010565,
    # N-terminal Carbamylation (+43.0058 Da, UNIMOD:5)
    "[UNIMOD:5]": 43.005814,
    "(+43.01)": 43.005814,
    "[+43.01]": 43.005814,
    "(+43.0058)": 43.005814,
    # N-terminal Ammonia Loss (-17.0265 Da, UNIMOD:385)
    "[UNIMOD:385]": -17.026549,
    "(-17.03)": -17.026549,
    "[-17.03]": -17.026549,
    "(-17.0265)": -17.026549,
}

# Standard 20 amino acids in a stable alphabetical order.
STANDARD_AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# Primary residue-level PTM tokens.
RESIDUE_PTM_TOKENS = [
    "M(ox)",
    "C(cam)",
    "N(deam)",
    "Q(deam)",
    "S(ph)",
    "T(ph)",
    "Y(ph)",
]

# N-terminal modification tokens.
N_TERM_MODIFICATIONS = [
    "(+42.01)",   # Acetylation [UNIMOD:1]
    "(+43.01)",   # Carbamylation [UNIMOD:5]
    "(-17.03)",   # Ammonia Loss [UNIMOD:385]
]

# Primary PTM tokens to include in extended vocabulary.
TARGET_PTM_TOKENS = RESIDUE_PTM_TOKENS + N_TERM_MODIFICATIONS

# Mapping from PTM token to unmodified parent amino acid for weight surgery.
PTM_PARENT_MAP = {
    "M(ox)": "M",
    "C(cam)": "C",
    "N(deam)": "N",
    "Q(deam)": "Q",
    "S(ph)": "S",
    "T(ph)": "T",
    "Y(ph)": "Y",
}

# Mapping from canonical token to standard UNIMOD format string.
CANONICAL_TO_UNIMOD_MAP = {
    "M(ox)": "M[UNIMOD:35]",
    "C(cam)": "C[UNIMOD:4]",
    "N(deam)": "N[UNIMOD:7]",
    "Q(deam)": "Q[UNIMOD:7]",
    "S(ph)": "S[UNIMOD:21]",
    "T(ph)": "T[UNIMOD:21]",
    "Y(ph)": "Y[UNIMOD:21]",
    "(+42.01)": "[UNIMOD:1]",
    "(+43.01)": "[UNIMOD:5]",
    "(-17.03)": "[UNIMOD:385]",
}

# Alternate delta and UNIMOD formatting aliases mapped to canonical token names.
PTM_ALIAS_MAP = {
    # Methionine oxidation (+15.9949 Da, UNIMOD:35)
    "M[UNIMOD:35]": "M(ox)",
    "M[15.9949]": "M(ox)",
    "M[15.99]": "M(ox)",
    "M(+15.99)": "M(ox)",
    "M(+15.995)": "M(ox)",
    "M[+16]": "M(ox)",
    # Cysteine carboxyamidomethylation (+57.0215 Da, UNIMOD:4)
    "C[UNIMOD:4]": "C(cam)",
    "C[57.0215]": "C(cam)",
    "C[57.02]": "C(cam)",
    "C(+57.02)": "C(cam)",
    "C(+57.021)": "C(cam)",
    "C[+57]": "C(cam)",
    # Asparagine deamidation (+0.9840 Da, UNIMOD:7)
    "N[UNIMOD:7]": "N(deam)",
    "N[0.9840]": "N(deam)",
    "N[0.98]": "N(deam)",
    "N(+0.98)": "N(deam)",
    "N(+.98)": "N(deam)",
    "N[+1]": "N(deam)",
    # Glutamine deamidation (+0.9840 Da, UNIMOD:7)
    "Q[UNIMOD:7]": "Q(deam)",
    "Q[0.9840]": "Q(deam)",
    "Q[0.98]": "Q(deam)",
    "Q(+0.98)": "Q(deam)",
    "Q(+.98)": "Q(deam)",
    "Q[+1]": "Q(deam)",
    # Serine phosphorylation (+79.9663 Da, UNIMOD:21)
    "S[UNIMOD:21]": "S(ph)",
    "S[79.9663]": "S(ph)",
    "S(p)": "S(ph)",
    "S(+79.97)": "S(ph)",
    "S(+79.966)": "S(ph)",
    "S[+80]": "S(ph)",
    # Threonine phosphorylation (+79.9663 Da, UNIMOD:21)
    "T[UNIMOD:21]": "T(ph)",
    "T[79.9663]": "T(ph)",
    "T(p)": "T(ph)",
    "T(+79.97)": "T(ph)",
    "T(+79.966)": "T(ph)",
    "T[+80]": "T(ph)",
    # Tyrosine phosphorylation (+79.9663 Da, UNIMOD:21)
    "Y[UNIMOD:21]": "Y(ph)",
    "Y[79.9663]": "Y(ph)",
    "Y(p)": "Y(ph)",
    "Y(+79.97)": "Y(ph)",
    "Y(+79.966)": "Y(ph)",
    "Y[+80]": "Y(ph)",
    # N-terminal Acetylation (+42.0106 Da, UNIMOD:1)
    "[UNIMOD:1]": "(+42.01)",
    "[42.0106]": "(+42.01)",
    "[+42.01]": "(+42.01)",
    "(+42.0106)": "(+42.01)",
    "[+42]": "(+42.01)",
    # N-terminal Carbamylation (+43.0058 Da, UNIMOD:5)
    "[UNIMOD:5]": "(+43.01)",
    "[43.0058]": "(+43.01)",
    "[+43.01]": "(+43.01)",
    "(+43.0058)": "(+43.01)",
    "[+43]": "(+43.01)",
    # N-terminal Ammonia Loss (-17.0265 Da, UNIMOD:385)
    "[UNIMOD:385]": "(-17.03)",
    "[-17.0265]": "(-17.03)",
    "[-17.03]": "(-17.03)",
    "(-17.0265)": "(-17.03)",
    "[-17]": "(-17.03)",
}

# Monoisotopic mass of a proton (Da).
M_H = 1.007276

# Monoisotopic mass of water (Da).
M_H2O = 18.010565
