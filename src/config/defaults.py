"""
Default hyperparameters for HC-PT (InstaDeepAI/ms_proteometools, ~2.7M spectra).

Tuned for large-scale training on a single A100 (40 GB) GCP VM.
InstaNovo uses ~95M parameters; these defaults scale the model up substantially
while remaining trainable on one high-end GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from data.lengths import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH


@dataclass(frozen=True)
class ModelDefaults:
    model_dim: int = 512
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_ff_dim: int = 1536   # tuned: 2048 → 1536 to stay in 40-50M param range
    decoder_blocks: int = 6
    decoder_heads: int = 8
    mlp_hidden_dim: int = 768    # tuned: 2048 → 768 (SwiGLU scales to 4× internally)
    max_charge: int = 6
    max_length: int = MAX_PEPTIDE_LENGTH
    min_length: int = MIN_PEPTIDE_LENGTH
    dropout: float = 0.1


@dataclass(frozen=True)
class DataDefaults:
    dataset_repo: str = "InstaDeepAI/ms_proteometools"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    top_k_peaks: int = 200  # InstaNovo n_peaks=200
    remove_precursor_peak: bool = True


@dataclass(frozen=True)
class TrainDefaults:
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_workers: int = 8
    device: str = "cuda"
    output_dir: str = "artifacts"
    # Generative validation metrics (de novo decode); eval_every>1 saves GPU time.
    eval_every: int = 2
    eval_max_batches: int = 64
    # 20 steps is sufficient for mid-training monitoring (full 50 only for final bench).
    inference_steps: int = 20
    noising_scheme: str = "mask"
    guidance_scale: float = 1.0
    compile: bool = False
    amp: bool = True
    length_noising: bool = True
    length_noising_prob: float = 0.1
    top_k_lengths: int = 3
    length_beam_alpha: float = 0.1


@dataclass(frozen=True)
class EvalDefaults:
    # 256 batch is safe with AMP + no_grad on A100 (4× the VRAM headroom vs training).
    batch_size: int = 256
    num_workers: int = 4
    inference_steps: int = 20
    noising_scheme: str = "mask"
    guidance_scale: float = 1.0
    top_k_lengths: int = 3
    length_beam_alpha: float = 0.1
    compile: bool = False
    amp: bool = True
    max_batches: int | None = None  # None = full split
    aa_mass_tolerance: float = 0.1
    prefix_mass_tolerance: float = 0.5


@dataclass(frozen=True)
class GCPDefaults:
    """
    Recommended Google Cloud VM for HC-PT training.

    Primary: a2-highgpu-1g  (1x NVIDIA A100 40GB, 12 vCPU, 85 GB RAM)
    Budget:  g2-standard-24 (1x NVIDIA L4 24GB, 24 vCPU, 96 GB RAM)
    """

    machine_type: str = "a2-highgpu-1g"
    accelerator_type: str = "nvidia-tesla-a100"
    accelerator_count: int = 1
    boot_disk_gb: int = 500
    zone: str = "us-central1-a"
    # Fallback if A100 quota unavailable.
    budget_machine_type: str = "g2-standard-24"
    budget_accelerator_type: str = "nvidia-l4"


@dataclass(frozen=True)
class ProjectDefaults:
    model: ModelDefaults = field(default_factory=ModelDefaults)
    data: DataDefaults = field(default_factory=DataDefaults)
    train: TrainDefaults = field(default_factory=TrainDefaults)
    eval: EvalDefaults = field(default_factory=EvalDefaults)
    gcp: GCPDefaults = field(default_factory=GCPDefaults)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = ProjectDefaults()
