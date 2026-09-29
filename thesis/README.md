# DFlowNovo AIMS Master's Thesis

This directory contains the complete LaTeX source code for the AIMS Structured Master's Research Project thesis, built strictly following the official [AIMS-Research/sm-project](https://github.com/AIMS-Research/sm-project.git) template.

## Document Metadata

- **Title:** DFlowNovo: Continuous-Time Discrete Flow Matching and Dynamic Knapsack Guidance for High-Throughput De Novo Peptide Sequencing
- **Author:** Joël Gédéon (`joel.gedeon@aims.ac.za`)
- **Institution:** African Institute for Mathematical Sciences (AIMS South Africa)
- **Supervisors:** Dr. Kevin Eloff and Prof. Ulrich Paquet (InstaDeep & AIMS South Africa)
- **Date:** September 2026

## Repository Structure

- [`master-document.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/master-document.tex): Master LaTeX document setting document geometry, preamble, packages, numbering, and structure.
- [`aimsessay.cls`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/aimsessay.cls): Official AIMS Research Project document class.
- [`myabbrvnat.bst`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/myabbrvnat.bst): Official AIMS Natbib bibliography style.
- [`abstract.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/abstract.tex): Abstract in English, Résumé in French (mother tongue / official language in Cameroon), and signed AIMS Declaration.
- [`acknowledgement.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/acknowledgement.tex): Professional academic and institutional acknowledgements.
- [`references.bib`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/references.bib): 29 verified BibTeX citations covering machine learning, generative flow matching, proteomics mass spectrometry, and peptide chemistry.
- **Chapters:**
  - [`chapter1.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter1.tex): Introduction to *De Novo* Peptide Sequencing and Mass Spectrometry.
  - [`chapter2.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter2.tex): Mathematical Foundations of Continuous-Time Discrete Flow Matching.
  - [`chapter3.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter3.tex): The DFlowNovo Framework: Architecture and Dynamic Knapsack Guidance.
  - [`chapter4.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter4.tex): Empirical Benchmark Evaluations and Quantitative Comparisons.
  - [`chapter5.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter5.tex): Mass Spectrometry Physics, Biological Fidelity and Error Dissection.
  - [`chapter6.tex`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/chapter6.tex): Conclusions and Future Perspectives.
- [`images/`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/thesis/images/): High-resolution 300 DPI vector and raster figures, AIMS logo, and official signature.

## Compilation Instructions

To compile the thesis into a PDF document, ensure TeX Live or MacTeX is installed, then run:

```bash
cd thesis
pdflatex master-document.tex
bibtex master-document
pdflatex master-document.tex
pdflatex master-document.tex
```

Or using `latexmk`:

```bash
cd thesis
latexmk -pdf master-document.tex
```
