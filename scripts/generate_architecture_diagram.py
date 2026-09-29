import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_architecture_diagram():
    # 16:9 Aspect Ratio, publication quality (300 DPI)
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Ultra-clean canvas background (Slate-50)
    canvas_bg = patches.Rectangle((0, 0), 16, 9, facecolor='#F8FAFC', edgecolor='none')
    ax.add_patch(canvas_bg)

    # -------------------------------------------------------------------------
    # MAIN HEADER
    # -------------------------------------------------------------------------
    ax.text(8.0, 8.65, "DFlowNovo: Architectural Framework & Iterative Decoding Pipeline",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0F172A',
            fontfamily='DejaVu Sans')
    ax.text(8.0, 8.35, "Discrete Flow Matching with Pre-LN Bidirectional Transformers & Exact GPU KnapsackDP Reachability",
            ha='center', va='center', fontsize=9.5, fontstyle='italic', color='#475569',
            fontfamily='DejaVu Sans')

    # Helper function for rendering elevated cards with subtle shadow & header band
    def draw_card(x, y, w, h, title, subtitle, theme_color, header_bg, shadow=True):
        if shadow:
            sh = FancyBboxPatch((x + 0.04, y - 0.04), w, h, boxstyle="round,pad=0.0,rounding_size=0.18",
                                facecolor='#E2E8F0', edgecolor='none', zorder=1)
            ax.add_patch(sh)
        # Main card body
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.18",
                              facecolor='#FFFFFF', edgecolor=theme_color, linewidth=1.5, zorder=2)
        ax.add_patch(card)
        # Header banner
        header_h = 0.54
        header = FancyBboxPatch((x, y + h - header_h), w, header_h,
                                boxstyle="round,pad=0.0,rounding_size=0.18",
                                facecolor=header_bg, edgecolor=theme_color, linewidth=1.2, zorder=3)
        ax.add_patch(header)
        # Header text
        ax.text(x + w / 2, y + h - 0.20, title,
                ha='center', va='center', fontsize=9.5, fontweight='bold', color=theme_color,
                fontfamily='DejaVu Sans', zorder=4)
        if subtitle:
            ax.text(x + w / 2, y + h - 0.39, subtitle,
                    ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#475569',
                    fontfamily='DejaVu Sans', zorder=4)

    # Helper for pill badges
    def draw_badge(x, y, w, h, text, bg_col, text_col, fontsize=7.2, bold=True, zorder=5):
        badge = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
                               facecolor=bg_col, edgecolor=text_col, linewidth=0.8, zorder=zorder)
        ax.add_patch(badge)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight,
                color=text_col, fontfamily='DejaVu Sans', zorder=zorder+1)

    # =========================================================================
    # COLUMN 1: INPUTS (x: 0.50 to 2.65, width: 2.15)
    # Baseline: y = 2.15, Topline: y = 7.85 (Height: 5.70)
    # =========================================================================
    spec_cx = 1.575

    # Card 1A: MS/MS Spectrum (y: 6.05 to 7.85, h: 1.80)
    draw_card(0.50, 6.05, 2.15, 1.80, "MS/MS SPECTRUM", "Observed Fragment Peaks", '#0F766E', '#F0FDFA')
    # Spectrum peak sticks
    ax.plot([spec_cx - 0.75, spec_cx + 0.75], [6.85, 6.85], color='#94A3B8', linewidth=1.2, zorder=4)
    peak_xs = [-0.6, -0.35, -0.1, 0.18, 0.42, 0.65]
    peak_hs = [0.28, 0.42, 0.20, 0.48, 0.32, 0.18]
    for px, ph in zip(peak_xs, peak_hs):
        ax.plot([spec_cx + px, spec_cx + px], [6.85, 6.85 + ph], color='#0D9488', linewidth=1.8, zorder=4)
        ax.scatter([spec_cx + px], [6.85 + ph], color='#0F766E', s=12, zorder=5)
    ax.text(spec_cx, 6.55, r"Peaks $(m_i, I_i)_{i=1}^M$", ha='center', va='center', fontsize=8.2,
            fontweight='bold', color='#0F172A', zorder=4)
    ax.text(spec_cx, 6.28, r"• $M \leq 500$ peaks, $\sqrt{I_i}$ scaled", ha='center', va='center',
            fontsize=7.4, color='#334155', zorder=4)

    # Card 1B: Precursor Metadata (y: 4.10 to 5.85, h: 1.75)
    draw_card(0.50, 4.10, 2.15, 1.75, "PRECURSOR DATA", "Physical Metadata", '#4338CA', '#EEF2FF')
    ax.text(spec_cx, 5.15, r"Neutral Mass $M_{\mathrm{prec}}$", ha='center', va='center', fontsize=8.2,
            fontweight='bold', color='#1E1B4B', zorder=4)
    ax.text(spec_cx, 4.85, r"Charge State $z \in \{2, 3, 4, 5\}$", ha='center', va='center', fontsize=8.2,
            fontweight='bold', color='#1E1B4B', zorder=4)
    ax.text(spec_cx, 4.50, "Global conditioning context", ha='center', va='center', fontsize=7.4,
            color='#475569', zorder=4)
    ax.text(spec_cx, 4.28, r"for Length Head & AdaLN-Zero", ha='center', va='center', fontsize=7.2,
            fontstyle='italic', color='#4338CA', zorder=4)

    # Card 1C: Noise Prior State (y: 2.15 to 3.90, h: 1.75)
    draw_card(0.50, 2.15, 2.15, 1.75, "PRIOR NOISE STATE", r"Initial Latent Sequence $x_0$", '#0369A1', '#F0F9FF')
    ax.text(spec_cx, 3.20, r"$x_0 \sim \mathrm{Cat}(1/S)$", ha='center', va='center', fontsize=8.8,
            fontweight='bold', color='#0C4A6E', zorder=4)
    ax.text(spec_cx, 2.85, r"Fully Masked: $[M]^L$", ha='center', va='center', fontsize=8.2,
            fontweight='bold', color='#0284C7', zorder=4)
    ax.text(spec_cx, 2.50, r"Vocabulary $\mathcal{V}$ ($|\mathcal{V}|=22$)", ha='center', va='center', fontsize=7.4,
            color='#334155', zorder=4)
    ax.text(spec_cx, 2.28, r"Starting state at flow time $t = 0$", ha='center', va='center', fontsize=7.2,
            fontstyle='italic', color='#0369A1', zorder=4)

    # =========================================================================
    # COLUMN 2: SPECTRUM ENCODER & LENGTH PREDICTOR (x: 3.25 to 6.35, w: 3.10)
    # Baseline: y = 2.15, Topline: y = 7.85
    # =========================================================================
    enc_cx = 4.80

    # Card 2A: Spectrum Encoder (y: 5.15 to 7.85, h: 2.70)
    draw_card(3.25, 5.15, 3.10, 2.70, "SPECTRUM ENCODER", "Bidirectional Pre-LN Transformer", '#0F766E', '#F0FDFA')
    # Section: Input
    draw_badge(enc_cx, 7.05, 2.00, 0.24, "INPUT: Peaks + Mass Duality", '#CCFBF1', '#0F766E', fontsize=7.2)
    ax.text(enc_cx, 6.78, r"$\Phi(m_i)$ and $\Phi(M_{\mathrm{prec}} - m_i + 2M_{\mathrm{H}})$", ha='center', va='center',
            fontsize=7.8, color='#0F172A', zorder=4)
    # Section: Core Architecture
    draw_badge(enc_cx, 6.45, 2.30, 0.24, "CORE: 6-Layer Transformer", '#CCFBF1', '#0F766E', fontsize=7.2)
    ax.text(enc_cx, 6.18, r"• $d = 512, h = 8$, Pre-LN + FlashAttention-2", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    ax.text(enc_cx, 5.92, r"• SwiGLU Feed-Forward Network ($d_{\mathrm{ffn}}=1376$)", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    # Section: Output
    draw_badge(enc_cx, 5.58, 2.20, 0.24, r"OUTPUT: Spectral Embeddings", '#0F766E', '#FFFFFF', fontsize=7.2)
    ax.text(enc_cx, 5.30, r"$H_{\mathrm{spec}} \in \mathbb{R}^{(M+1) \times d}$ and $[CLS]$ token",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0F766E', zorder=4)

    # Card 2B: Length Predictor (y: 2.15 to 4.85, h: 2.70)
    draw_card(3.25, 2.15, 3.10, 2.70, "LENGTH PREDICTOR", "Auxiliary Peptide Length Head", '#059669', '#ECFDF5')
    # Section: Input
    draw_badge(enc_cx, 4.05, 2.20, 0.24, r"INPUT: Conditioning Summary", '#D1FAE5', '#065F46', fontsize=7.2)
    ax.text(enc_cx, 3.78, r"Global $[CLS]$ token from $H_{\mathrm{spec}}$ + $[M_{\mathrm{prec}}, z]$",
            ha='center', va='center', fontsize=7.6, color='#0F172A', zorder=4)
    # Section: Core Architecture
    draw_badge(enc_cx, 3.45, 2.10, 0.24, "CORE: 2-Layer MLP", '#D1FAE5', '#065F46', fontsize=7.2)
    ax.text(enc_cx, 3.18, r"• Linear $(d \to 256)$ + GELU + Linear $(256 \to 28)$", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    ax.text(enc_cx, 2.92, r"• Cross-Entropy Length Prior Loss $\mathcal{L}_{\mathrm{len}}$", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    # Section: Output
    draw_badge(enc_cx, 2.58, 2.30, 0.24, r"OUTPUT: Predicted Length $L$", '#059669', '#FFFFFF', fontsize=7.2)
    ax.text(enc_cx, 2.30, r"Top Candidates $\mathcal{L}_{\mathrm{cand}} \sim \mathrm{TopK}(p_{\mathrm{len}}, B=3)$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#065F46', zorder=4)

    # =========================================================================
    # COLUMN 3: DISCRETE FLOW DECODER (x: 6.95 to 10.15, w: 3.20)
    # Baseline: y = 2.15, Topline: y = 7.85 (Height: 5.70)
    # =========================================================================
    draw_card(6.95, 2.15, 3.20, 5.70, "DISCRETE FLOW DECODER", "Bidirectional Discrete Flow Matching", '#1D4ED8', '#EFF6FF')
    dec_cx = 8.55

    # Section: Inputs & Conditioning
    draw_badge(dec_cx, 7.00, 2.40, 0.26, "INPUTS & CONDITIONING", '#DBEAFE', '#1E40AF', fontsize=7.4)
    ax.text(dec_cx, 6.68, r"• Intermediate Sequence State: $x_t \in \mathcal{V}^L$", ha='center', va='center',
            fontsize=7.8, color='#0F172A', zorder=4)
    ax.text(dec_cx, 6.38, r"• Continuous Flow Time: $t \in [0, 1]$", ha='center', va='center',
            fontsize=7.8, color='#0F172A', zorder=4)
    ax.text(dec_cx, 6.08, r"• Precursor Context: $(M_{\mathrm{prec}}, z)$ via AdaLN-Zero", ha='center', va='center',
            fontsize=7.8, color='#0F172A', zorder=4)

    # Section: Core Architecture
    draw_badge(dec_cx, 5.60, 2.50, 0.26, "CORE: 6-Block Transformer", '#DBEAFE', '#1E40AF', fontsize=7.4)
    ax.text(dec_cx, 5.26, "• Bidirectional Self-Attention over sequence tokens", ha='center', va='center',
            fontsize=7.6, color='#334155', zorder=4)
    ax.text(dec_cx, 4.96, r"• Full Cross-Attention into Spectral Context $H_{\mathrm{spec}}$", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#1D4ED8', zorder=4)
    ax.text(dec_cx, 4.66, r"• AdaLN-Zero Modulation $(\gamma_l, \beta_l, \alpha_l)$ zero-initialized", ha='center', va='center',
            fontsize=7.6, color='#334155', zorder=4)
    ax.text(dec_cx, 4.36, r"• SwiGLU Feed-Forward ($d=512, d_{\mathrm{ffn}}=1376$)", ha='center', va='center',
            fontsize=7.6, color='#334155', zorder=4)
    ax.text(dec_cx, 4.02, r"• Continuous-Time Flow Objective (Campbell et al. 2024):", ha='center', va='center',
            fontsize=7.4, fontstyle='italic', color='#475569', zorder=4)
    ax.text(dec_cx, 3.72, r"$\mathcal{L}_{\mathrm{DFM}} = -\sum_{d: x_t^d=[M]} \log p_{1|t}^\theta(x_1^d \mid x_t, t, \mathcal{S})$",
            ha='center', va='center', fontsize=7.8, color='#1E40AF', zorder=4)

    # Section: Output
    draw_badge(dec_cx, 3.20, 2.60, 0.28, r"OUTPUT: Transition Rates $R_t^\theta$", '#1D4ED8', '#FFFFFF', fontsize=7.4)
    ax.text(dec_cx, 2.78, r"Categorical Rate Logits $\mathbf{Z} \in \mathbb{R}^{L \times 22}$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1E3A8A', zorder=4)
    ax.text(dec_cx, 2.50, r"Governs probability velocity toward true sequence $x_1$",
            ha='center', va='center', fontsize=7.4, fontstyle='italic', color='#334155', zorder=4)

    # =========================================================================
    # COLUMN 4: ITERATIVE GENERATION & KNAPSACK DP LOOP (x: 10.75 to 13.95, w: 3.20)
    # Baseline: y = 2.15, Topline: y = 7.85 (Height: 5.70)
    # =========================================================================
    draw_card(10.75, 2.15, 3.20, 5.70, "ITERATIVE KNAPSACK-DP LOOP", "Euler Trajectory & Exact Mass Reachability", '#D97706', '#FFFBEB')
    loop_cx = 12.35

    # Section: 1. Reverse Euler Step
    draw_badge(loop_cx, 7.00, 2.60, 0.26, "1. REVERSE EULER STEP", '#FEF3C7', '#B45309', fontsize=7.4)
    ax.text(loop_cx, 6.68, r"• Discrete CTMC integration: $t_k \to t_{k+1}$ ($K=20$ steps)", ha='center', va='center',
            fontsize=7.6, color='#0F172A', zorder=4)
    ax.text(loop_cx, 6.38, r"• Velocity / dynamic unmasking rate: $\lambda(t) \cdot \Delta t$", ha='center', va='center',
            fontsize=7.6, color='#0F172A', zorder=4)

    # Section: 2. Exact GPU Knapsack DP Filter
    draw_badge(loop_cx, 5.85, 2.70, 0.26, "2. GPU KNAPSACK DP FILTER", '#FEF3C7', '#B45309', fontsize=7.4)
    ax.text(loop_cx, 5.52, r"• Precomputed reachability table $M_{\mathrm{valid}}$ ($\Delta m = 0.02\,\mathrm{Da}$)",
            ha='center', va='center', fontsize=7.6, fontweight='bold', color='#92400E', zorder=4)
    ax.text(loop_cx, 5.22, r"• Bounded prefix & suffix precursor mass budget", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    ax.text(loop_cx, 4.88, r"• Exact $\mathcal{O}(1)$ Dynamic Logit Pruning:", ha='center', va='center',
            fontsize=7.6, fontweight='bold', color='#B45309', zorder=4)
    ax.text(loop_cx, 4.58, r"$\tilde{Z}_{d, a} = Z_{d, a} + \log M_{\mathrm{valid}}(a)$", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#D97706', zorder=4)
    ax.text(loop_cx, 4.26, "Prunes unreachable amino acids; guarantees 0.0 ppm mass fit", ha='center', va='center',
            fontsize=7.2, fontstyle='italic', color='#78350F', zorder=4)

    # Section: 3. State Update & Jump
    draw_badge(loop_cx, 3.75, 2.60, 0.26, "3. STATE UPDATE & JUMP", '#FEF3C7', '#B45309', fontsize=7.4)
    ax.text(loop_cx, 3.40, r"• Sample transition: $x_{t+\Delta t} \sim \mathrm{Cat}(\operatorname{softmax}(\tilde{Z}))^{\lambda_t}$",
            ha='center', va='center', fontsize=7.6, color='#0F172A', zorder=4)
    ax.text(loop_cx, 3.08, r"• Progressively resolves masked latents to peptide", ha='center', va='center',
            fontsize=7.5, color='#334155', zorder=4)
    ax.text(loop_cx, 2.75, r"• Loop iterates until continuous flow time $t = 1$", ha='center', va='center',
            fontsize=7.6, fontweight='bold', color='#B45309', zorder=4)

    # =========================================================================
    # COLUMN 5: FINAL OUTPUT (x: 14.35 to 15.65, w: 1.30)
    # Baseline: y = 2.15, Topline: y = 7.85 (Height: 5.70)
    # =========================================================================
    draw_card(14.35, 2.15, 1.30, 5.70, "FINAL OUTPUT", "Valid Peptide", '#059669', '#ECFDF5')
    out_cx = 15.00

    ax.text(out_cx, 7.02, "Peptide Sequence", ha='center', va='center', fontsize=7.8,
            fontweight='bold', color='#065F46', zorder=4)
    
    # Stylized amino acid residue tokens
    residues = ["P", "E", "P", "T", "I", "D", "E"]
    aa_y_start = 6.62
    for idx, aa in enumerate(residues):
        y_pos = aa_y_start - idx * 0.38
        draw_badge(out_cx, y_pos, 0.70, 0.28, aa, '#D1FAE5', '#047857', fontsize=8.5, bold=True)

    ax.text(out_cx, 3.70, r"Final: $\hat{Y}^*$", ha='center', va='center', fontsize=9.2,
            fontweight='bold', color='#065F46', zorder=4)
    
    # Validation indicators
    draw_badge(out_cx, 3.20, 1.10, 0.26, "✓ Strict Mass", '#ECFDF5', '#059669', fontsize=7.0)
    ax.text(out_cx, 2.92, r"$\Delta M \approx 0.0\,\mathrm{ppm}$", ha='center', va='center',
            fontsize=7.2, color='#065F46', zorder=4)
    draw_badge(out_cx, 2.55, 1.10, 0.26, "✓ Calibrated", '#ECFDF5', '#059669', fontsize=7.0)

    # =========================================================================
    # ARROWS & CONNECTORS
    # =========================================================================
    # Arrow 1: Spectrum Peaks (1A) -> Spectrum Encoder (2A)
    arr_1a_2a = FancyArrowPatch((2.65, 6.95), (3.25, 6.95), arrowstyle='simple,head_width=6,head_length=8',
                                facecolor='#0D9488', edgecolor='#0F766E', linewidth=0.8, zorder=6)
    ax.add_patch(arr_1a_2a)
    ax.text(2.95, 7.15, "Peaks", ha='center', va='center', fontsize=7.4, fontweight='bold',
            color='#0F766E', zorder=6)

    # Arrow 2: Precursor Data (1B) -> Spectrum Encoder (2A) [Complementary mass]
    arr_1b_2a = FancyArrowPatch((2.65, 5.25), (3.25, 5.55), connectionstyle="arc3,rad=-0.12",
                                arrowstyle='simple,head_width=5,head_length=7',
                                facecolor='#6366F1', edgecolor='#4338CA', linewidth=0.8, zorder=6)
    ax.add_patch(arr_1b_2a)

    # Arrow 3: Precursor Data (1B) -> Length Predictor (2B)
    arr_1b_2b = FancyArrowPatch((2.65, 4.45), (3.25, 4.15), connectionstyle="arc3,rad=0.12",
                                arrowstyle='simple,head_width=5,head_length=7',
                                facecolor='#6366F1', edgecolor='#4338CA', linewidth=0.8, zorder=6)
    ax.add_patch(arr_1b_2b)
    ax.text(2.95, 4.45, r"$M_{\mathrm{prec}}, z$", ha='center', va='center', fontsize=7.2,
            fontweight='bold', color='#4338CA', zorder=6)

    # Arrow 4: Spectrum Encoder (2A) -> Length Predictor (2B) [CLS token]
    arr_enc_len = FancyArrowPatch((4.80, 5.15), (4.80, 4.85),
                                  arrowstyle='simple,head_width=5,head_length=7',
                                  facecolor='#0F766E', edgecolor='#115E59', linewidth=0.8, zorder=6)
    ax.add_patch(arr_enc_len)
    ax.text(5.15, 5.00, r"$[CLS]$", ha='left', va='center', fontsize=7.5,
            fontweight='bold', color='#0F766E', zorder=6)

    # Arrow 5: Spectrum Encoder (2A) -> Discrete Flow Decoder (3) [H_spec Cross-Attention]
    arr_enc_dec = FancyArrowPatch((6.35, 6.50), (6.95, 6.50),
                                  arrowstyle='simple,head_width=7,head_length=9',
                                  facecolor='#0D9488', edgecolor='#0F766E', linewidth=1.0, zorder=6)
    ax.add_patch(arr_enc_dec)
    ax.text(6.65, 6.75, r"$H_{\mathrm{spec}}$", ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='#0F766E', zorder=6)

    # Arrow 6: Length Predictor (2B) -> Decoder Sequence Length L
    arr_len_dec = FancyArrowPatch((6.35, 3.50), (6.95, 3.50),
                                  arrowstyle='simple,head_width=6,head_length=8',
                                  facecolor='#059669', edgecolor='#047857', linewidth=0.8, zorder=6)
    ax.add_patch(arr_len_dec)
    ax.text(6.65, 3.72, r"Length $L$", ha='center', va='center', fontsize=7.4,
            fontweight='bold', color='#047857', zorder=6)

    # Arrow 7: Decoder (3) -> KnapsackDP & Euler Loop (4) [Velocity Logits R_t]
    arr_dec_loop = FancyArrowPatch((10.15, 5.00), (10.75, 5.00),
                                   arrowstyle='simple,head_width=7,head_length=9',
                                   facecolor='#2563EB', edgecolor='#1D4ED8', linewidth=1.0, zorder=6)
    ax.add_patch(arr_dec_loop)
    ax.text(10.45, 5.25, r"$R_t^\theta$", ha='center', va='center', fontsize=9.0,
            fontweight='bold', color='#1D4ED8', zorder=6)

    # =========================================================================
    # DEDICATED BOTTOM CORRIDOR (y: 0.60 to 2.15): INITIAL STATE & EULER LOOP
    # =========================================================================
    # Arrow 8: Prior Noise State (1C) -> Initial state input at t=0
    # Runs out of 1C bottom, across open corridor at y=1.60, into Decoder at x=7.50, y=2.15
    arr_p1 = (1.575, 2.15)
    arr_p2 = (1.575, 1.60)
    arr_p3 = (7.65, 1.60)
    arr_p4 = (7.65, 2.15)

    ax.plot([arr_p1[0], arr_p2[0]], [arr_p1[1], arr_p2[1]], color='#0284C7', linewidth=1.5, zorder=6)
    ax.plot([arr_p2[0], arr_p3[0]], [arr_p2[1], arr_p3[1]], color='#0284C7', linewidth=1.5, zorder=6)
    arr_init_up = FancyArrowPatch((arr_p3[0], arr_p3[1]), arr_p4,
                                  arrowstyle='simple,head_width=5,head_length=7',
                                  facecolor='#0284C7', edgecolor='#0369A1', linewidth=0.8, zorder=6)
    ax.add_patch(arr_init_up)
    draw_badge(4.40, 1.60, 2.80, 0.30,
               r"Initial State $x_0 \sim \mathrm{Cat}(1/S)$  at  $t = 0$",
               '#F0F9FF', '#0284C7', fontsize=7.4, bold=True, zorder=7)

    # Arrow 9: CYCLIC FEEDBACK LOOP (Euler integration update x_{t+dt} -> Decoder x_t)
    # Curves down out of Column 4, runs along open corridor at y=1.05, curves up into Column 3
    loop_start_x = 12.00
    loop_entry_x = 8.85
    loop_y_corridor = 1.05

    ax.plot([loop_start_x, loop_start_x], [2.15, loop_y_corridor], color='#D97706', linewidth=2.0, zorder=6)
    ax.plot([loop_start_x, loop_entry_x], [loop_y_corridor, loop_y_corridor], color='#D97706', linewidth=2.0, zorder=6)
    arr_loop_up = FancyArrowPatch((loop_entry_x, loop_y_corridor), (loop_entry_x, 2.15),
                                  arrowstyle='simple,head_width=6,head_length=8',
                                  facecolor='#D97706', edgecolor='#B45309', linewidth=1.0, zorder=6)
    ax.add_patch(arr_loop_up)

    # Feedback Loop Pill Badge
    draw_badge(10.45, 1.05, 3.50, 0.36,
               r"$\circlearrowleft$  Euler Trajectory Loop: $x_{t+\Delta t} \to x_t$  ($k = 1, \dots, K$ steps)",
               '#FFFBEB', '#B45309', fontsize=7.8, bold=True, zorder=7)

    # Arrow 10: Knapsack Loop (4) -> Output (5) [Exit at t=1]
    arr_loop_out = FancyArrowPatch((13.95, 5.00), (14.35, 5.00),
                                   arrowstyle='simple,head_width=6,head_length=8',
                                   facecolor='#059669', edgecolor='#047857', linewidth=0.8, zorder=6)
    ax.add_patch(arr_loop_out)
    ax.text(14.15, 5.25, r"$t = 1$", ha='center', va='center', fontsize=7.8,
            fontweight='bold', color='#059669', zorder=6)

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    out_thesis = Path("thesis/images/dflow_architecture_pipeline.png")
    out_docs = Path("docs/figures/dflow_architecture_pipeline.png")
    out_pres = Path("presentation/images/dflow_architecture_pipeline.png")
    out_artifact = Path("/home/joelgedeon_aims_ac_za/.gemini/antigravity-cli/brain/ee9cf031-b0c5-4740-b963-40c81c13ea78/dflow_architecture_pipeline.png")

    out_thesis.parent.mkdir(parents=True, exist_ok=True)
    out_docs.parent.mkdir(parents=True, exist_ok=True)
    out_pres.parent.mkdir(parents=True, exist_ok=True)
    out_artifact.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_thesis, dpi=300, bbox_inches='tight')
    fig.savefig(out_docs, dpi=300, bbox_inches='tight')
    fig.savefig(out_pres, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')

    print(f"[✓] Successfully generated redesigned architecture diagram:")
    print(f"    - {out_thesis}")
    print(f"    - {out_docs}")
    print(f"    - {out_pres}")
    print(f"    - {out_artifact}")
    plt.close(fig)

if __name__ == "__main__":
    create_architecture_diagram()
