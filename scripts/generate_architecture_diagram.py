import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_architecture_diagram():
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Background canvas
    bg = patches.Rectangle((0, 0), 16, 12, facecolor='#F8FAFC', edgecolor='none')
    ax.add_patch(bg)

    # Title header
    ax.text(8.0, 11.6, "DFlowNovo: Architectural Framework and Iterative Decoding Pipeline", 
            ha='center', va='center', fontsize=17, fontweight='bold', color='#0F172A')
    ax.text(8.0, 11.3, "Continuous-Time Discrete Flow Matching with Exact KnapsackDP Reachability and Bayesian Beam Guidance", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569')

    # =========================================================================
    # PART A: NEURAL NETWORK MODEL ARCHITECTURE (TRAINING & FORWARD PASS)
    # =========================================================================
    part_a_box = FancyBboxPatch((0.5, 6.0), 15.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                                facecolor='#FFFFFF', edgecolor='#2563EB', linewidth=2.2, linestyle='-')
    ax.add_patch(part_a_box)

    # Part A Badge
    part_a_badge = FancyBboxPatch((0.8, 10.6), 6.6, 0.45, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#1E40AF', edgecolor='none')
    ax.add_patch(part_a_badge)
    ax.text(4.1, 10.82, "PART A: NEURAL NETWORK MODEL ARCHITECTURE (FORWARD PASS)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF')

    # Sub-block A1: Raw Spectrum Featurization
    a1_box = FancyBboxPatch((0.8, 6.4), 3.2, 3.9, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#EFF6FF', edgecolor='#93C5FD', linewidth=1.2)
    ax.add_patch(a1_box)
    ax.text(2.4, 9.95, "1. Spectrum Featurizer", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E3A8A')
    
    # Details in A1
    ax.text(2.4, 9.5, "Input: Tandem Spectrum S", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(2.4, 9.15, r"• Observed Peaks: $(m_i, I_i)_{i=1}^M$ ($M \leq 500$)", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(2.4, 8.7, "• Complementary Mass Duality:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(2.4, 8.35, r"$m_{\mathrm{comp}, i} = M_{\mathrm{prec}} - m_i + 2 M_{\mathrm{H}}$", ha='center', va='center', fontsize=8.5, color='#2563EB')
    ax.text(2.4, 7.85, "• Sinusoidal Projections:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(2.4, 7.5, r"$\Phi(m_i) \in \mathbb{R}^{d/2}$, $\Phi(m_{\mathrm{comp}, i}) \in \mathbb{R}^{d/2}$", ha='center', va='center', fontsize=8, color='#334155')
    ax.text(2.4, 7.0, r"• Square-root Intensity: $\sqrt{I_i} / \sum \sqrt{I}$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(2.4, 6.6, r"Output: Peak Tokens $E_{\mathrm{peak}} \in \mathbb{R}^{M \times d}$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E40AF')

    # Arrow A1 -> A2
    arr_a1_a2 = FancyArrowPatch((4.05, 8.35), (4.55, 8.35), arrowstyle='simple,head_width=6,head_length=8',
                                facecolor='#3B82F6', edgecolor='#1D4ED8', linewidth=0.8)
    ax.add_patch(arr_a1_a2)

    # Sub-block A2: Bidirectional Spectrum Transformer Encoder
    a2_box = FancyBboxPatch((4.6, 6.4), 3.2, 3.9, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#F0FDF4', edgecolor='#86EFAC', linewidth=1.2)
    ax.add_patch(a2_box)
    ax.text(6.2, 9.95, "2. Spectrum Encoder", ha='center', va='center', fontsize=11, fontweight='bold', color='#14532D')
    ax.text(6.2, 9.5, "Bidirectional Transformer", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(6.2, 9.15, r"• $N_{\mathrm{enc}} = 6$ Layers, $d = 512$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(6.2, 8.75, "• Pre-LayerNorm & FlashAttention-2", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(6.2, 8.35, "• Multi-Head Self-Attention ($h=8$)", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(6.2, 7.95, r"• SwiGLU FFN ($d_{\mathrm{ffn}} = 1376$)", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(6.2, 7.45, "• Global [CLS] + Peak Tokens", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(6.2, 6.9, r"Output: $H_{\mathrm{spec}} \in \mathbb{R}^{(M+1) \times d}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#15803D')

    # Length Head Branch from A2
    a2_len_box = FancyBboxPatch((4.8, 6.48), 2.8, 0.42, boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor='#DCFCE7', edgecolor='#4ADE80', linewidth=1.0)
    ax.add_patch(a2_len_box)
    ax.text(6.2, 6.69, r"Length Head MLP: $p_{\mathrm{len}} \in \Delta^{27}$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#166534')

    # Arrow A2 -> A3 (H_spec to Decoder)
    arr_a2_a3 = FancyArrowPatch((7.85, 8.35), (8.35, 8.35), arrowstyle='simple,head_width=6,head_length=8',
                                facecolor='#10B981', edgecolor='#047857', linewidth=0.8)
    ax.add_patch(arr_a2_a3)

    # Sub-block A3: AdaLN-Zero Conditioning Generator
    a3_box = FancyBboxPatch((8.4, 8.0), 2.7, 2.3, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#FFFBEB', edgecolor='#FCD34D', linewidth=1.2)
    ax.add_patch(a3_box)
    ax.text(9.75, 9.95, "3. AdaLN-Zero Modulator", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#78350F')
    ax.text(9.75, 9.55, r"Inputs: $(t, M_{\mathrm{prec}}, z)$", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(9.75, 9.15, r"• Flow time $t \in [0, 1]$", ha='center', va='center', fontsize=8, color='#334155')
    ax.text(9.75, 8.8, r"• Precursor Mass $M_{\mathrm{prec}}$", ha='center', va='center', fontsize=8, color='#334155')
    ax.text(9.75, 8.45, r"• Precursor Charge $z \in \{2, 3, 4, 5\}$", ha='center', va='center', fontsize=8, color='#334155')
    ax.text(9.75, 8.15, r"Yields $(\gamma_l, \beta_l, \alpha_l)$ zero-init", ha='center', va='center', fontsize=8, fontweight='bold', color='#B45309')

    # Sub-block A4: Bidirectional Discrete Flow Decoder
    a4_box = FancyBboxPatch((11.4, 6.4), 3.8, 3.9, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#FAF5FF', edgecolor='#D8B4FE', linewidth=1.2)
    ax.add_patch(a4_box)
    ax.text(13.3, 9.95, "4. Bidirectional Flow Decoder", ha='center', va='center', fontsize=11, fontweight='bold', color='#581C87')
    ax.text(13.3, 9.55, r"$N_{\mathrm{dec}} = 6$ Layers with AdaLN-Zero", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(13.3, 9.15, r"• Input: Corrupted Sequence $x_t \in \mathcal{V}^L$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(13.3, 8.75, r"• Cross-Attention into $H_{\mathrm{spec}}$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(13.3, 8.35, "• Adaptive Scaling & Gating:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(13.3, 8.0, r"$h^{(l)} = h + \alpha \odot \mathrm{Block}(\mathrm{LN}(h)(1+\gamma) + \beta)$", ha='center', va='center', fontsize=7.8, color='#7E22CE')
    ax.text(13.3, 7.5, r"• Output: Unmasked Logits $\mathbf{Z} \in \mathbb{R}^{L \times S}$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#6B21A8')
    ax.text(13.3, 7.05, "Training Loss (Campbell et al. 2024):", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(13.3, 6.65, r"$\mathcal{L}_{\mathrm{DFM}} = - \sum_{d: x_t^d=[M]} \log p_{1|t}^\theta(x_1^d \mid x_t, t, \mathcal{S})$", ha='center', va='center', fontsize=8.0, color='#9333EA')

    # Arrow A3 -> A4
    arr_a3_a4 = FancyArrowPatch((11.15, 8.8), (11.35, 8.8), arrowstyle='simple,head_width=5,head_length=6',
                                facecolor='#F59E0B', edgecolor='#D97706', linewidth=0.8)
    ax.add_patch(arr_a3_a4)

    # =========================================================================
    # PART B: MASS-GUIDED ITERATIVE DECODING & FILTERING PIPELINE (INFERENCE)
    # =========================================================================
    part_b_box = FancyBboxPatch((0.5, 0.6), 15.0, 4.9, boxstyle="round,pad=0.2,rounding_size=0.4",
                                facecolor='#FFFFFF', edgecolor='#059669', linewidth=2.2, linestyle='-')
    ax.add_patch(part_b_box)

    # Part B Badge
    part_b_badge = FancyBboxPatch((0.8, 5.05), 7.8, 0.45, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#065F46', edgecolor='none')
    ax.add_patch(part_b_badge)
    ax.text(4.7, 5.27, "PART B: MASS-GUIDED ITERATIVE DECODING & FILTERING PIPELINE (INFERENCE)", 
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#FFFFFF')

    # Sub-block B1: Length Beam & Prior Initialization
    b1_box = FancyBboxPatch((0.8, 1.0), 3.2, 3.8, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#ECFDF5', edgecolor='#A7F3D0', linewidth=1.2)
    ax.add_patch(b1_box)
    ax.text(2.4, 4.45, "1. Beam Prior Setup", ha='center', va='center', fontsize=11, fontweight='bold', color='#064E3B')
    ax.text(2.4, 4.05, r"From Length Head $p_{\mathrm{len}}$:", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(2.4, 3.7, "• Extract Top-B Candidates", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(2.4, 3.35, r"$\mathcal{L}_{\mathrm{cand}} = \mathrm{TopK}(p_{\mathrm{len}}, B=3)$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#047857')
    ax.text(2.4, 2.9, "• Initialize Fully Masked Latents:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(2.4, 2.55, r"$x_0^{(b)} = [ [M], [M], \dots, [M] ] \in \mathcal{V}^{L_b}$", ha='center', va='center', fontsize=8.5, color='#065F46')
    ax.text(2.4, 2.05, "• Uniform Categorical Prior:", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(2.4, 1.7, r"$p_0(x_0) = \mathrm{Cat}(1/S)$ on $\mathcal{V}^{L_b}$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(2.4, 1.25, r"Parallel Batch: $B \times \mathrm{Beam}$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#059669')

    # Inter-panel connector: Length Head (A2) -> Beam Prior Setup (B1)
    arr_len_to_b1 = FancyArrowPatch((5.8, 6.48), (2.8, 4.8), connectionstyle="arc3,rad=-0.2",
                                    arrowstyle='simple,head_width=5,head_length=7',
                                    facecolor='#059669', edgecolor='#065F46', linewidth=0.8)
    ax.add_patch(arr_len_to_b1)

    # Sub-block B2: Precomputed GPU KnapsackDP
    b2_box = FancyBboxPatch((4.3, 1.0), 3.3, 3.8, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#FFF7ED', edgecolor='#FFEDD5', linewidth=1.2)
    ax.add_patch(b2_box)
    ax.text(5.95, 4.45, "2. GPU KnapsackDP Table", ha='center', va='center', fontsize=11, fontweight='bold', color='#7C2D12')
    ax.text(5.95, 4.05, "Exact DP Reachability Table:", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(5.95, 3.7, r"$T[k, m] = \bigvee_{a} T[k-1, m - w_a]$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#C2410C')
    ax.text(5.95, 3.25, r"• Discretization $\Delta m = 0.02\,\mathrm{Da}$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(5.95, 2.85, r"• Bins $B_{\max} = 250,000$ ($M \leq 5000$ Da)", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(5.95, 2.45, "• GPU 1D Max-Pooling (W=101):", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(5.95, 2.1, r"$T_{\mathrm{pool}}[k, b] = \max_{|j-b| \leq 50} T[k, j]$", ha='center', va='center', fontsize=8.5, color='#EA580C')
    ax.text(5.95, 1.6, "Guarantees Precursor Mass Fit", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#9A3412')
    ax.text(5.95, 1.25, r"Precomputed in $< 450\,\mathrm{ms}$ (1-time)", ha='center', va='center', fontsize=8, fontstyle='italic', color='#64748B')

    # Sub-block B3: Reverse Euler Integration Loop with Dynamic Pruning
    b3_box = FancyBboxPatch((8.0, 1.0), 3.9, 3.8, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#FEF2F2', edgecolor='#FECACA', linewidth=1.2)
    ax.add_patch(b3_box)
    ax.text(9.95, 4.45, "3. Iterative Reverse Euler Loop", ha='center', va='center', fontsize=11, fontweight='bold', color='#991B1B')
    ax.text(9.95, 4.05, "20-Step Cosine Flow Integration:", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(9.95, 3.7, r"For step $k = 0, \dots, K-1$ ($t_k \to t_{k+1}$):", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(9.95, 3.3, r"1. Call Decoder: $\mathbf{Z} = \mathrm{Dec}(x_t, t_k, H_{\mathrm{spec}})$", ha='center', va='center', fontsize=8.0, color='#991B1B')
    ax.text(9.95, 2.85, r"2. $\mathcal{O}(1)$ KnapsackDP Pruning:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(9.95, 2.5, r"$\tilde{Z}_{d, a} = Z_{d, a} \text{ if Valid}(a) \text{ else } -\infty$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#DC2626')
    ax.text(9.95, 2.05, "3. Dynamic Euler Unmasking Jump:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(9.95, 1.7, r"$x_{t+\Delta t}^d \sim \mathrm{Cat}(p_{1|t}^d)$ with prob $\lambda_t$", ha='center', va='center', fontsize=8.0, color='#7F1D1D')
    ax.text(9.95, 1.25, "Enforces Strict 0.0 ppm Intact Mass", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#B91C1C')

    # Connection: Knapsack Table (B2) -> Pruning in B3
    arr_b2_b3 = FancyArrowPatch((7.65, 2.5), (7.95, 2.5), arrowstyle='simple,head_width=5,head_length=6',
                                facecolor='#EA580C', edgecolor='#C2410C', linewidth=0.8)
    ax.add_patch(arr_b2_b3)

    # Inter-panel connector: Decoder (A4) -> Reverse Euler Loop (B3)
    arr_a4_to_b3 = FancyArrowPatch((12.5, 6.35), (10.5, 4.85), connectionstyle="arc3,rad=-0.25",
                                   arrowstyle='simple,head_width=6,head_length=8',
                                   facecolor='#7C3AED', edgecolor='#5B21B6', linewidth=0.8)
    ax.add_patch(arr_a4_to_b3)
    ax.text(12.2, 5.5, r"Forward Pass Logits $\mathbf{Z}$", ha='center', va='center', fontsize=8, fontweight='bold', color='#6D28D9',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#F3E8FF', edgecolor='#C4B5FD', linewidth=0.8))

    # Sub-block B4: Bayesian Candidate Scoring & Re-ranking
    b4_box = FancyBboxPatch((12.2, 1.0), 3.0, 3.8, boxstyle="round,pad=0.15,rounding_size=0.25",
                            facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2)
    ax.add_patch(b4_box)
    ax.text(13.7, 4.45, "4. Bayesian Re-ranking", ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')
    ax.text(13.7, 4.05, "Joint Posterior Objective:", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax.text(13.7, 3.65, r"$\mathrm{Score}(Y) = \log p(L \mid \mathcal{S})$", ha='center', va='center', fontsize=8.2, color='#1E293B')
    ax.text(13.7, 3.3, r"$+ \frac{1}{L} \sum \log p(Y_i \mid \mathcal{S}, L)$", ha='center', va='center', fontsize=8.2, color='#1E293B')
    ax.text(13.7, 2.9, r"$- \alpha \cdot |\Delta M_{\mathrm{prec}}| / M$", ha='center', va='center', fontsize=8.2, color='#DC2626')
    ax.text(13.7, 2.5, r"$+ \beta \cdot \mathrm{FragMatch}(Y, \mathcal{S})$", ha='center', va='center', fontsize=8.2, color='#059669')
    ax.text(13.7, 2.1, r"$+ \gamma \cdot \mathrm{TrypsinPrior}(y_1)$", ha='center', va='center', fontsize=8.2, color='#2563EB')
    ax.text(13.7, 1.6, r"Optimal Output: $\hat{Y}^*$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(13.7, 1.25, "Consensus Strict Match", ha='center', va='center', fontsize=8.5, color='#475569')

    # Arrow B3 -> B4
    arr_b3_b4 = FancyArrowPatch((11.95, 2.9), (12.15, 2.9), arrowstyle='simple,head_width=5,head_length=6',
                                facecolor='#475569', edgecolor='#334155', linewidth=0.8)
    ax.add_patch(arr_b3_b4)

    # Save to disk
    out_thesis = Path("thesis/images/dflow_architecture_pipeline.png")
    out_docs = Path("docs/figures/dflow_architecture_pipeline.png")
    out_artifact = Path("/home/joelgedeon_aims_ac_za/.gemini/antigravity-cli/brain/ee9cf031-b0c5-4740-b963-40c81c13ea78/dflow_architecture_pipeline.png")

    fig.savefig(out_thesis, dpi=300, bbox_inches='tight')
    fig.savefig(out_docs, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
    print(f"Successfully generated architecture diagram at {out_thesis}")
    plt.close(fig)

if __name__ == "__main__":
    create_architecture_diagram()
