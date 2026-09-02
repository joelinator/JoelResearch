# Etat d'Avancement du Projet: DFLowNovo

**Chercheur:** Joel Gedeon  
**Institution:** AIMS South Africa  
**Sujet:** De Novo Peptide Sequencing via Discrete Flow Matching (DFM)

---

## 1. Resume du Projet

Le projet DFLowNovo a pour objectif la prediction non-autoregressive de sequences peptidiques a partir de spectres de spectrometrie de masse en tandem (MS/MS). Contrairement aux approches autoregressives sequentielles (ex. InstaNovo), ce modele genere l'integralite de la sequence simultanement via un processus d'appariement de flux discret (Discrete Flow Matching).

---

## 2. Synthese des Travaux Realises

### 2.1. Traitement des Donnees et Ingestion Physique
* Ingestion du jeu de donnees **ProteomeTools** (~2.7 millions de spectres a haute energie de collision).
* Normalisation des intensites par racine carree et filtrage des 200 pics les plus intenses.
* Suppression automatique du pic precurseur non fragmente.
* Calcul et injection explicite des pics complementaires $m/z$ pour assister l'attention dans la correlation des paires d'ions fragments $b$ et $y$:
  $$(m/z)_{comp, i} = M_{prec} - (m/z)_i$$

### 2.2. Architecture du Modele DFLowNovo
* **Encodeur Spectral:** 6 couches d'attention multi-tetes (MHSA) traitant les coordonnees $m/z$, $m/z$ complementaires et intensites projetees.
* **Predicteur de Longueur:** Tete MLP predisant la distribution de longueur $p(L \mid \mathcal{S})$ pour $L \in [1, 30]$.
* **Decodeur a Flux Discret:** 
  * Normalisation adaptative des couches (**AdaLN**) pour conditionner le decodeur sur le temps de diffusion $t$, la masse precurseur $M_{prec}$, la charge $z$, et la longueur $L$.
  * Remplacement des plongements sinusoidaux de longueur par des plongements discrets explicites (`nn.Embedding`).
  * Activation **SwiGLU** (facteur d'expansion $8/3$) pour une meilleure convergence et stabilite.
  * **Guidage sans classificateur (CFG):** Entrainement avec masquage aleatoire du conditionneur spectral ($p = 0.1$) pour permettre une extrapolation des logits a l'inference.

### 2.3. Fonctions de Perte et Regularisations
* Formulation d'une perte multi-tache programmee:
  $$\mathcal{L}_{total} = \mathcal{L}_{decoder} + \lambda(e)\mathcal{L}_{length} + \gamma(e)\mathcal{L}_{mass}$$
* **Regularisation par bruitage de longueur (Length Noising):** Perturbation aleatoire de la longueur passee au decodeur ($\pm 1$, probabilite 0.10) durant l'entrainement pour rendre le decodeur robuste aux imprecisions du predicteur de longueur. Masquage automatique de la perte de masse sur les batchs bruites.
* **Penalisation de masse Huber:** Contrainte physique sur la somme des masses residuelles moyennes pour forcer la coherence avec la masse du precurseur.

### 2.4. Inference et Decodage par Faisceau Top-k de Longueurs (Top-k Length Beam Decoding)
* Decodage parallelise sur les $K$ longueurs les plus probables ($K=3$ par defaut) en un seul passage GPU vectorise.
* Score d'evaluation conjoint:
  $$\text{Score}(\hat{Y}^{(k)}) = -\mathcal{H}(\hat{Y}^{(k)} \mid \mathcal{S}) - \alpha \cdot \left\vert{} \sum_{i=1}^{L_k} m(\hat{Y}_i^{(k)}) - (M_{prec} - M_{H_2O}) \right\vert{}$$
* Selection optimale maximisant la vraisemblance spectrale et minimisant l'ecart de masse.

### 2.5. Optimisation Technique et Deploiement Cloud
* Entrainement multi-GPU avec PyTorch Lightning, accumulation de gradients (batch effectif 512), precision mixte `bfloat16`, et compilation JIT `torch.compile`.
* Journalisation double via **CSV** et **TensorBoard** pour un suivi en temps reel.
* Scripts d'automatisation GCP pour la creation de VM A100, la synchronisation du code, l'execution en arriere-plan sous tmux et le suivi a distance.

---

## 3. Objectifs et Prochaines Etapes

* **Lancement de l'entrainement complet:** Execution des 30 epoques sur VM A100 GCP sur les 2.7M de spectres.
* **Evaluation comparative (Benchmarking):** Calcul des metriques de rappel et precision au niveau acide amine et peptide complet face aux modeles autoregressifs de reference (InstaNovo, Pi-PrimeNovo).
* **Ajustement des hyperparametres d'inference:** Optimisation de l'echelle de guidage CFG $s$ et du coefficient de penalite de masse $\alpha$.
* **Analyse d'interpretabilite:** Etude des cartes d'attention de l'encodeur spectral pour quantifier l'identification des paires d'ions $b/y$.

---

## 4. Articles Etudies et References

* InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments | Nature Machine Intelligence
* [2406.04843] Variational Flow Matching for Graph Generation
* [2402.04997] Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design
* Regressor-guided Diffusion Model for De Novo Peptide Sequencing with Explicit Mass Control
* Pi-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing | Nature Communications
