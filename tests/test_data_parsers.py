import pytest
from src.data.data import parse_mgf_sequence


def test_parse_mgf_sequence_from_dict_seq():
    params = {"seq": "PEPTIDER"}
    assert parse_mgf_sequence(params) == "PEPTIDER"


def test_parse_mgf_sequence_from_dict_sequence():
    params = {"sequence": "SAMPLEPEPTIDE"}
    assert parse_mgf_sequence(params) == "SAMPLEPEPTIDE"


def test_parse_mgf_sequence_from_dict_title_seq():
    params = {"title": "Scan 1234 SEQ=ALBU[+16]MIN Charge=2"}
    assert parse_mgf_sequence(params) == "ALBU[+16]MIN"


def test_parse_mgf_sequence_from_raw_string():
    raw_mgf_block = """BEGIN IONS
TITLE=Spectrum 42
PEPMASS=650.32
CHARGE=2+
SEQ=LVEALYLVCGER
SCANS=42
END IONS"""
    assert parse_mgf_sequence(raw_mgf_block) == "LVEALYLVCGER"


def test_parse_mgf_sequence_missing():
    params = {"title": "Unknown spectrum without sequence", "charge": "2+"}
    assert parse_mgf_sequence(params) is None
    assert parse_mgf_sequence(None) is None
