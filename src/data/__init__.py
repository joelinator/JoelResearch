from .constants import AA_MASSES_DICT, M_H, STANDARD_AMINO_ACIDS

__all__ = [
    "AA_MASSES_DICT",
    "M_H",
    "STANDARD_AMINO_ACIDS",
    "MAX_PEPTIDE_LENGTH",
    "MIN_PEPTIDE_LENGTH",
    "NUM_LENGTH_CLASSES",
    "SpectrumDataSet",
    "build_dataloader",
    "build_vocabulary",
    "apply_length_padding",
    "class_to_length",
    "clamp_length",
    "decoder_output_size",
    "decoder_output_token_ids",
    "get_aa_masses",
    "get_dataset",
    "get_output_aa_masses",
    "length_to_active_mask",
    "length_to_class",
    "spectrum_collate",
    "validate_peptide_length",
]

_LENGTH_NAMES = {
    "MAX_PEPTIDE_LENGTH",
    "MIN_PEPTIDE_LENGTH",
    "NUM_LENGTH_CLASSES",
    "apply_length_padding",
    "class_to_length",
    "clamp_length",
    "length_to_active_mask",
    "length_to_class",
    "validate_peptide_length",
}

_DATA_NAMES = {
    "SpectrumDataSet",
    "build_dataloader",
    "build_vocabulary",
    "decoder_output_size",
    "decoder_output_token_ids",
    "get_aa_masses",
    "get_dataset",
    "get_output_aa_masses",
    "spectrum_collate",
}


def __getattr__(name: str):
    if name in _LENGTH_NAMES:
        from . import lengths as lengths_module

        return getattr(lengths_module, name)
    if name in _DATA_NAMES:
        from . import data as data_module

        return getattr(data_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
