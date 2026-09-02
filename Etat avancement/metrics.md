# Evaluation Metrics

Evaluating de novo peptide sequencing requires specialized metrics. Standard NLP metrics (like BLEU or raw Accuracy) are insufficient because they ignore the underlying physics of mass spectrometry. 

For instance, the amino acids **Leucine (L)** and **Isoleucine (I)** have the exact same chemical formula and mass (113.08406 Da). A mass spectrometer cannot distinguish them. If the ground truth is `PEPTIDEI` and the model predicts `PEPTIDEL`, the prediction is physically perfect and must be scored as 100% correct.

## 1. Amino Acid Precision and Recall
To calculate precision and recall at the amino acid level, we align the predicted sequence against the ground truth sequence using a mass-based Dynamic Programming (DP) algorithm.

*   **Prefix Mass Alignment:** Instead of comparing characters at index $i$, we compute the cumulative prefix mass up to index $i$ for both strings. 
*   **Tolerance:** Two amino acids are considered a match if the difference between their prefix masses is less than $\Delta = 0.5$ Da, and the mass of the amino acids themselves differ by less than $\delta = 0.1$ Da.
*   **Precision:** $P = \frac{N_{matched}}{N_{predicted}}$
*   **Recall:** $R = \frac{N_{matched}}{N_{target}}$

This DP algorithm matches the exact benchmark implementations used in **InstaNovo** and **Casanovo**, ensuring our results are strictly comparable in literature.

## 2. Exact Peptide Recall
Peptide recall is a strict, sequence-level metric. A predicted sequence is considered correct if and only if:
1. Every amino acid is successfully matched to the target sequence via the mass-based DP alignment.
2. The total mass of the predicted sequence falls within a specific tolerance of the precursor mass.

Because non-autoregressive models predict lengths independently, they sometimes predict sequences that are missing one amino acid, resulting in 0% Peptide Recall even if Amino Acid Precision is 95%. To alleviate this during inference, advanced decoding strategies like **Knapsack Beam Search** (or mass-directed prefix pruning) can be applied.

## 3. Teacher-Forced (Proxy) Metrics
Because the generative reverse diffusion process requires 20-50 forward passes per spectrum, computing generative metrics continuously during training is prohibitively slow.
To track progress during training epochs, we compute **Teacher-Forced Metrics**:
*   `token_accuracy`: Standard categorical accuracy on the un-noised tokens (like a BERT Masked-Language-Modeling metric).
*   `length_accuracy`: The categorical accuracy of the Length Predictor head.
*   `exact_peptide_accuracy`: 100% if all tokens match, 0% otherwise.

While these do not perfectly correlate with true generative Peptide Recall, they are highly reliable proxies that compute in a single forward pass, allowing us to accurately monitor overfitting and convergence.
