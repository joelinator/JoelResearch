# Dataset and Preprocessing

## 1. Dataset Origin
We utilize the **ProteomeTools** dataset, specifically the High-Collision Energy (HC) subset hosted on Hugging Face by InstaDeep (`InstaDeepAI/ms_proteometools`). 
ProteomeTools contains highly curated synthetic peptides, which guarantees extremely high confidence in the ground truth sequences, unlike empirically observed datasets where false discovery rates (FDR) can introduce noisy labels. 
The dataset scale is roughly **2.7 million spectra** for training.

## 2. Dataset Fields
Each entry in the parquet files contains the raw physical observations of a mass spectrometry event:
* `mz_array`: An array of float values representing the mass-to-charge ratios of the fragmented ions.
* `intensity_array`: The corresponding intensity (abundance) of each ion in the `mz_array`.
* `precursor_mz`: The mass-to-charge ratio of the unfragmented intact peptide.
* `precursor_charge`: The ionization charge $z$ (typically between $1+$ and $6+$).
* `sequence`: The ground truth amino acid string (e.g., `"PEPTIDE"`).

## 3. Physical Preprocessing
Before feeding the data to the model, several transformations are applied to map physical quantities into normalized tensors.

### Precursor Neutral Mass Calculation
The model needs the neutral mass of the intact peptide $M_{prec}$ rather than the $m/z$ ratio. We compute this using the physics formula for protonated ions:
$$ M_{prec} = (\text{precursor\_mz} \times z) - (z \times M_{H^+}) $$
where $M_{H^+} \approx 1.007276$ Da is the mass of a proton.

### Peak Filtering and Normalization
A raw spectrum can contain thousands of peaks, many of which are chemical noise.
1. **Filtering:** We select only the top $k = 200$ peaks sorted by intensity.
2. **Intensity Normalization:** The intensities vary over several orders of magnitude. We normalize them by dividing by the maximum intensity in the spectrum, scaling them to $[0, 1]$:
   $$ I_{normalized}^{(i)} = \frac{I^{(i)}}{\max_{j} I^{(j)}} $$

### MZ-Complementary Calculation
For every retained peak $m/z_i$, we compute its complementary peak (assuming charge 1 for fragment ions, a standard simplification):
$$ m/z_{comp, i} = M_{prec} - m/z_i $$
This allows the Transformer to easily attend to $b$-ion / $y$-ion pairs.

## 4. Text Tokenization
The ground truth sequence is converted into discrete tokens.
* **Vocabulary:** 20 standard amino acids (e.g., `A`, `C`, `D`, ...).
* **Special Tokens:** `<pad>` (index 20) for batch padding, and `<mask_token>` (index 21) which is used as the absorbing state during the discrete diffusion process.
* **Padding:** Sequences are padded to a maximum length (e.g., 30). An `active_mask` is created to ignore `<pad>` tokens during the loss calculation.
