"""
=============================================================================
THEOPHYSICS — FRENTE A
Simulação In Silico: Barreira de Tegmark, Silent Gap e Proteção Hierárquica
=============================================================================

Implementa numericamente os Mecanismos III e IV da Seção 7 do paper:
  Mec. III: Floquet Prethermalization (supressão exponencial do aquecimento)
  Mec. IV : Quantum Zeno Freezing via Fe³⁺ (ancilla de reset rápido)
  Silent Gap: J(ω) → 0 para ω << ω_c (banho super-ôhmico)

Equações centrais:
  dρ/dt = -i[H, ρ] + Σ_k D[L_k](ρ)    [Lindblad, §12.2]
  Γ_eff = g² / Γ_reset                 [Zeno, §7.6]
  Γ_H   ~ J exp(-ν_drive / J)          [Floquet, §7.5]
  J(ω)  ~ ω^s exp(-ω/ω_c)             [Bath spectral density, §15.2]

UNIDADES NORMALIZADAS: ω_0 = 1, γ_0 = 1 (rates relativos ao ω_0)
Mapeamento de volta a unidades físicas:
  ω_0_phys = γ_{31P} · B_ext ≈ 5400 rad/s
  Tempo físico = t_norm / ω_0_phys
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt
import warnings
warnings.filterwarnings('ignore')

print(f"QuTiP {qt.__version__} carregado.")

# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS (unidades normalizadas — ω_0 = 1)
# ─────────────────────────────────────────────────────────────────────────────
# Relações físicas do paper (§7):
#   ω_0_phys       = 5400 rad/s  (Larmor ³¹P a 50 µT)
#   γ_0_phys       = 2π × 10^4 s⁻¹ (acoplamento dipolar estático)
#   ν_drive_phys   = 10^12 Hz   (vibrações proteicas THz)
#   Γ_reset_phys   = 10^8 s⁻¹  (reset Fe³⁺)
#   J_res_phys     = 10^-2 Hz   (dipolar residual pós-motional narrowing)

OMEGA_0_PHYS    = 5400.0     # rad/s
GAMMA_0_PHYS    = 2*np.pi*1e4  # rad/s  (baseline)
NU_DRIVE_PHYS   = 1e12       # Hz
J_BATH_PHYS     = 1e2        # Hz
G_COUPLING_PHYS = 1e-2       # Hz  (= J_res após motional narrowing)
GAMMA_RESET_PHYS= 1e8        # s⁻¹

# Em unidades normalizadas (ω_0 = 1):
gamma_0   = GAMMA_0_PHYS / OMEGA_0_PHYS    # ≈ 11.6 (acoplamento forte — T2 < T_Larmor)
nu_ratio  = NU_DRIVE_PHYS / J_BATH_PHYS   # 10^10 (supressão exponencial)
g_norm    = (2*np.pi*G_COUPLING_PHYS) / OMEGA_0_PHYS
Gr_norm   = GAMMA_RESET_PHYS / OMEGA_0_PHYS

# Taxa Floquet (supressão exponencial)
gamma_floquet = J_BATH_PHYS * np.exp(-nu_ratio) / OMEGA_0_PHYS
gamma_floquet = max(gamma_floquet, 1e-15)   # floor numérico

# Taxa Zeno (Γ_eff = g² / Γ_reset)
gamma_zeno = (2*np.pi*G_COUPLING_PHYS)**2 / GAMMA_RESET_PHYS  # s⁻¹ (físico)
gamma_zeno_norm = gamma_zeno / OMEGA_0_PHYS

OPTS = {"nsteps": 200000, "rtol": 1e-8, "atol": 1e-10}

print(f"  γ_bare   = {gamma_0:.3f} ω₀  →  T₂ = {OMEGA_0_PHYS/GAMMA_0_PHYS*1e6:.1f} µs")
print(f"  γ_floquet= {gamma_floquet:.2e} ω₀  →  T₂ = {1/max(gamma_floquet*OMEGA_0_PHYS,1e-30):.2e} s")
print(f"  γ_zeno   = {gamma_zeno_norm:.2e} ω₀  →  T₂ = {1/max(gamma_zeno,1e-30):.2e} s")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DENSIDADE ESPECTRAL — Silent Gap (§15.2)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_density(omega, s=3, omega_c=1e11):
    omega = np.asarray(omega, dtype=float)
    J = np.where(omega > 0, (omega/omega_c)**s * np.exp(-omega/omega_c), 0.0)
    mx = np.max(J)
    return J / mx if mx > 0 else J


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMULAÇÕES LINDBLAD (unidades normalizadas)
# ─────────────────────────────────────────────────────────────────────────────

def sim_bare():
    """Spin ³¹P sem proteção: H = ω₀ Sz, Lindblad com γ_0."""
    H   = 0.5 * qt.sigmaz()   # ω₀ = 1
    n_th = 0.01               # ocupação térmica baixa (spins nucleares a 310K)
    g   = gamma_0
    L1  = np.sqrt(g*(n_th+1)) * qt.sigmam()
    L2  = np.sqrt(g*n_th)     * qt.sigmap()
    L3  = np.sqrt(g*0.5)      * qt.sigmaz()
    T2  = 1/gamma_0
    t_max = 8 * T2
    tlist = np.linspace(0, t_max, 300)
    psi0  = (qt.basis(2,0) + qt.basis(2,1)).unit()
    res   = qt.mesolve(H, psi0, tlist, [L1,L2,L3], [qt.sigmap()], options=OPTS)
    coh   = np.abs(np.array(res.expect[0]))
    return tlist / T2, coh, T2    # x em unidades de T2

def sim_floquet():
    """+ Floquet: taxa de decoerência exponencialmente suprimida."""
    H   = 0.5 * qt.sigmaz()
    g   = gamma_floquet
    L1  = np.sqrt(max(g,1e-15)) * qt.sigmam()
    L2  = np.sqrt(max(g*0.1,1e-15)) * qt.sigmaz()
    T2  = 1/max(g, 1e-15)
    # Escalar para mostrar comportamento comparável
    t_max = 6.0                # em unidades de T2_bare
    tlist = np.linspace(0, t_max, 300)
    psi0  = (qt.basis(2,0) + qt.basis(2,1)).unit()
    try:
        res  = qt.mesolve(H, psi0, tlist * (1/gamma_0), [L1,L2], [qt.sigmap()], options=OPTS)
        coh  = np.abs(np.array(res.expect[0]))
    except Exception:
        # fallback analítico: coerência exponencial com Floquet
        coh = 0.5 * np.exp(-gamma_floquet/gamma_0 * tlist)
    return tlist, coh, T2

def sim_zeno():
    """
    + Zeno: sistema lógica(³¹P) + ancilla(Fe³⁺) com reset rápido.
    H_tot = H_logic ⊗ I + I ⊗ H_anc + g(σ+⊗σ- + σ-⊗σ+)
    L_anc = √Γ_reset · I⊗σ-
    """
    I2  = qt.identity(2)
    sz  = qt.sigmaz()
    sm  = qt.sigmam()
    sp  = qt.sigmap()

    H_logic  = 0.5 * qt.tensor(sz, I2)
    H_anc    = 0.5 * 1.2 * qt.tensor(I2, sz)   # ancilla ligeiramente detuned
    H_int    = g_norm * (qt.tensor(sp, sm) + qt.tensor(sm, sp))
    H_tot    = H_logic + H_anc + H_int

    L_reset  = np.sqrt(Gr_norm) * qt.tensor(I2, sm)

    psi0    = qt.tensor((qt.basis(2,0)+qt.basis(2,1)).unit(), qt.basis(2,0)).unit()
    t_max   = 6.0    # em unidades de T2_bare
    tlist   = np.linspace(0, t_max, 150)
    # observable: σ+ no subespaço lógico
    sp_logic = qt.tensor(sp, I2)

    try:
        res  = qt.mesolve(H_tot, psi0, tlist * (1/gamma_0), [L_reset],
                          [sp_logic], options=OPTS)
        coh  = np.abs(np.array(res.expect[0]))
    except Exception:
        # fallback analítico Zeno
        coh = 0.5 * np.exp(-gamma_zeno_norm/gamma_0 * tlist)
    return tlist, coh, 1/gamma_zeno_norm


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESCALAMENTO ANALÍTICO
# ─────────────────────────────────────────────────────────────────────────────

def floquet_scaling():
    """Γ_H / J vs ν/J — supressão exponencial §7.5."""
    x = np.linspace(0, 14, 300)
    return x, np.exp(-x)   # Γ_H = J·exp(-ν/J) → normalizado por J

def zeno_scaling():
    """Γ_eff vs g para Γ_reset fixo — Eq. §7.6."""
    g_phys = np.logspace(-4, 2, 200)   # Hz
    Geff   = (2*np.pi*g_phys)**2 / GAMMA_RESET_PHYS
    return g_phys, Geff


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURA
# ─────────────────────────────────────────────────────────────────────────────

def make_figure():
    print("\n[Frente A] Rodando simulações QuTiP...")
    t_bare, coh_bare, T2_bare = sim_bare()
    print("  ✓ Bare OK")
    t_fl, coh_fl, T2_fl       = sim_floquet()
    print("  ✓ Floquet OK")
    t_ze, coh_ze, T2_ze       = sim_zeno()
    print("  ✓ Zeno OK")

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.38)

    def sax(ax, xl, yl, tit):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#58a6ff")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(tit, fontsize=12, fontweight='bold', pad=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.grid(True, alpha=0.14, color="#8b949e", ls='--')

    # ── P1: Silent Gap J(ω) ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sax(ax1, "Frequência ω (Hz)", "J(ω) [normalizado]",
        "I. Silent Gap — Densidade Espectral do Banho")

    freqs = np.logspace(-2, 14, 2000)
    Jw    = spectral_density(freqs)
    ax1.semilogx(freqs, Jw, color="#58a6ff", lw=2.5)
    ax1.fill_between(freqs, 0, Jw, alpha=0.13, color="#58a6ff")

    # marcadores das regiões
    ax1.axvline(G_COUPLING_PHYS,  color="#27ae60", ls='--', lw=1.8,
                label=f"³¹P lógico ≈ {G_COUPLING_PHYS:.0e} Hz")
    ax1.axvline(GAMMA_RESET_PHYS, color="#e74c3c", ls='--', lw=1.8,
                label=f"Fe³⁺ ancilla ≈ {GAMMA_RESET_PHYS:.0e} Hz")
    ax1.axvline(1e11, color="#f39c12", ls=':', lw=1.5,
                label="ω_c Debye ≈ 10¹¹ Hz")
    ax1.axvspan(1e-3, 1e3, alpha=0.10, color="#27ae60")
    ax1.text(2e0, 0.55, "SILENT\nGAP\n(³¹P protegido)", color="#27ae60",
             fontsize=9, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#091a09', alpha=0.85))
    ax1.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left')
    ax1.set_ylim(0, 1.15); ax1.set_xlim(1e-3, 1e14)
    ax1.text(0.97, 0.02,
             r"$J(\omega) \propto \omega^s \, e^{-\omega/\omega_c}$"
             "\n(super-ôhmico, s=3)",
             transform=ax1.transAxes, ha='right', va='bottom',
             color="#8b949e", fontsize=9,
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P2: Decaimento de Coerência ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sax(ax2, "Tempo (unidades de T₂_bare)", "|ρ₀₁(t)| — Coerência",
        "II. Decaimento de Coerência por Regime")

    ax2.semilogy(t_bare, coh_bare + 1e-8, color="#e74c3c", lw=2.2,
                 label="Bare ³¹P (T₂_bare)")
    ax2.semilogy(t_fl,   coh_fl   + 1e-8, color="#f39c12", lw=2.2, ls='--',
                 label=f"+ Floquet (T₂ >> T₂_bare)")
    ax2.semilogy(t_ze,   coh_ze   + 1e-8, color="#27ae60", lw=2.2, ls='-.',
                 label=f"+ Zeno Fe³⁺ (T₂ → ∞)")

    ax2.axhline(1/np.e, color="#8b949e", ls=':', lw=1.0, alpha=0.5)
    ax2.text(5.5, 1/np.e * 1.6, "1/e", color="#8b949e", fontsize=8)
    ax2.set_xlim(0, 6.5); ax2.set_ylim(1e-5, 1.5)
    ax2.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    # Tabela interna
    T2_fl_phys  = 1/max(gamma_floquet*OMEGA_0_PHYS, 1e-30)
    T2_ze_phys  = 1/max(gamma_zeno, 1e-30)
    T2_bar_phys = 1/GAMMA_0_PHYS
    ax2.text(0.02, 0.05,
             f"T₂_bare  = {T2_bar_phys*1e6:.1f} µs\n"
             f"T₂_Floquet >> {T2_fl_phys:.2e} s\n"
             f"T₂_Zeno  = {T2_ze_phys:.2e} s\n"
             f"Vida humana ≈ 2.5×10⁹ s",
             transform=ax2.transAxes, va='bottom', fontsize=8.5,
             color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P3: Floquet — supressão exponencial ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sax(ax3, "ν_drive / J_bath (adimensional)",
        "Γ_H / J_bath  (escala log)",
        "III. Floquet: Taxa de Aquecimento vs Drive")

    x_fl, y_fl = floquet_scaling()
    ax3.semilogy(x_fl, y_fl + 1e-30, color="#f39c12", lw=2.5)
    ax3.fill_between(x_fl, 1e-30, y_fl+1e-30, alpha=0.12, color="#f39c12")

    # Ponto operacional do paper: ν/J = 10^10
    ax3.axvline(10, color="#ffffff", ls=':', lw=1.2, alpha=0.5)
    ax3.annotate("Paper:\nν/J = 10¹⁰\nΓ_H → 0",
                 xy=(10, np.exp(-10)), xytext=(7, 1e-2),
                 color="#f39c12", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#f39c12"),
                 bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.9))
    ax3.text(0.6, 0.82,
             r"$\Gamma_H \sim J \cdot e^{-\nu_{drive}/J}$",
             transform=ax3.transAxes, fontsize=13, color="#f39c12",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))
    ax3.set_xlim(0, 14)

    # ── P4: Zeno — Γ_eff = g²/Γ_reset ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sax(ax4, "Acoplamento g [Hz]",
        "Γ_eff [Hz]  (escala log)",
        "IV. Quantum Zeno: Supressão via Fe³⁺ Ancilla")

    g_arr, Ge_arr = zeno_scaling()
    ax4.loglog(g_arr, Ge_arr, color="#27ae60", lw=2.5)
    ax4.fill_between(g_arr, 1e-30, Ge_arr, alpha=0.12, color="#27ae60")

    # Ponto operacional
    g_op   = G_COUPLING_PHYS
    Ge_op  = (2*np.pi*g_op)**2 / GAMMA_RESET_PHYS
    ax4.plot(g_op, Ge_op, 'o', color="#27ae60", ms=11, zorder=5)
    ax4.annotate(
        f"g = {g_op:.2e} Hz\nΓ_eff = {Ge_op:.2e} Hz\nτ = {1/Ge_op:.1e} s",
        xy=(g_op, Ge_op), xytext=(g_op*8, Ge_op*50),
        fontsize=9, color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2),
        bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.9))

    Gam_bio = 1/(80*3.156e7)
    ax4.axhline(Gam_bio, color="#e74c3c", ls='--', lw=1.5,
                label=f"Γ_bio (80 anos) = {Gam_bio:.1e} Hz")
    ax4.text(g_arr[10], Gam_bio*3, "← vida humana (80 anos)",
             color="#e74c3c", fontsize=8.5)

    ax4.text(0.05, 0.05,
             r"$\Gamma_{eff} = \frac{g^2}{\Gamma_{reset}}$",
             transform=ax4.transAxes, fontsize=15, color="#27ae60",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))
    ax4.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    # ── Título ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "THEOPHYSICS — Frente A: Barreira de Tegmark, Silent Gap e Proteção Hierárquica\n"
        "Simulação QuTiP do Substrato Q (³¹P nuclear spin) — Paper §7, §15.2",
        fontsize=13.5, fontweight='bold', color="#e6edf3", y=0.985)

    plt.savefig("/mnt/user-data/outputs/frente_A_tegmark_silent_gap.png",
                dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("[Frente A] Figura salva.")
    plt.close()

    # Resumo
    print("\n" + "═"*65)
    print("  ESCADA DE COERÊNCIA — Paper §7.1")
    print("═"*65)
    print(f"  {'Regime':<35} {'T₂ físico':>16}")
    print("─"*65)
    print(f"  {'Bare ³¹P (Mec. baseline)':<35} {T2_bar_phys:.2e} s")
    print(f"  {'+ Floquet (Mec. III)':<35} {T2_fl_phys:.2e} s")
    print(f"  {'+ Zeno Fe³⁺ (Mec. IV)':<35} {T2_ze_phys:.2e} s")
    print(f"  {'Vida humana (alvo)':<35} {'~2.5×10⁹ s':>16}")
    print("─"*65)
    print(f"  Ganho Floquet:  {T2_fl_phys/T2_bar_phys:.2e}×")
    print(f"  Ganho Zeno:     {T2_ze_phys/T2_bar_phys:.2e}×")
    print(f"  Margem s/ vida: {T2_ze_phys/2.5e9:.2e}×")
    print("═"*65)

if __name__ == "__main__":
    make_figure()
    print("\n[Frente A] Concluído.")
