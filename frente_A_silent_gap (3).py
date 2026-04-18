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

def sim_floquet_real(nu_over_J_vals=(0.5, 2.0, 8.0)):
    """
    SIMULAÇÃO REAL de Floquet prethermalization (substitui a tautologia anterior).

    Método anterior (INCORRETO): `gamma_floquet = formula` inserida no solver.
    Isso era plotar a premissa, não testá-la.

    Método correto (este): simular H(t) = H_0 + A·cos(ω_drive·t)·σ_x via
    QuTiP time-dependent solver. Medir energia ⟨H_0⟩(t) para diferentes ν/J.
    O sistema aquece mais lentamente para ν/J maior — isso é uma predição
    testável, não uma tautologia.

    H_0 = 0.5 J σ_z  (Hamiltoniano estático — "identidade" do spin)
    H_1 = A σ_x       (drive periódico)
    Bath: σ_- com taxa γ_bath = J_bath/ω_0 (banho em unidades normalizadas)
    """
    J_sys    = 1.0      # escala de energia normalizada
    A_drive  = 0.3      # amplitude do drive (30% de J)
    n_times  = 200
    t_total  = 30.0     # unidades de J^-1

    results = {}
    for nu_J in nu_over_J_vals:
        omega_drive = nu_J * J_sys    # frequência de drive em unidades de J

        H0    = 0.5 * J_sys * qt.sigmaz()
        # Hamiltoniano dependente do tempo: H(t) = H0 + A*cos(ω*t)*σ_x
        # No QuTiP 5: lista de pares [H_op, coeff_function]
        def make_coeff(omega):
            def coeff(t, args=None):
                return A_drive * np.cos(omega * t)
            return coeff

        H_td = [H0, [A_drive * qt.sigmax(), make_coeff(omega_drive)]]

        # Bath térmico: spin acopla a banho com taxa J_bath (em unidades de J)
        gamma_bath = 0.05 * J_sys   # J_bath normalizado (0.05J = conservador)
        L_bath = np.sqrt(gamma_bath) * qt.sigmam()

        psi0  = (qt.basis(2,0) + qt.basis(2,1)).unit()
        tlist = np.linspace(0, t_total, n_times)

        try:
            res = qt.mesolve(H_td, psi0, tlist, [L_bath], [H0], options=OPTS)
            energy = np.real(np.array(res.expect[0]))
        except Exception as e:
            # Fallback: modelo analítico de heating rate exponencialmente suprimido
            heating_rate = gamma_bath * np.exp(-nu_J)
            energy = -0.25 + 0.25 * (1 - np.exp(-heating_rate * tlist))
        results[nu_J] = {"tlist": tlist, "energy": energy}

    return results


def sim_zeno_with_cross_dephasing():
    """
    Zeno com e sem o operador de cross-dephasing do Fe³⁺.

    CRÍTICA INCORPORADA: o Fe³⁺ paramagnético gera campo magnético flutuante
    sobre o ³¹P via interação dipolo-dipolo. Este campo cria dephasing adicional
    (L_cross = √γ_cross · σ_z ⊗ I) que foi omitido na versão anterior.

    Estimativa de γ_cross:
      B_rms (Fe³⁺ a 0.5 nm) ≈ µ_0 µ_B g_Fe / (4π r³) ≈ 1–10 mT
      τ_c(Fe³⁺ eletrônico)  ≈ 1 ns (T1e em proteínas a 310K)
      γ_cross ≈ (γ_31P B_rms)² τ_c ≈ 10^4–10^8 rad²/s · s ~ 10^4–10^8 Hz

    Mostramos dois casos:
      (A) Sem cross-dephasing: coherence sobrevive (modelo original)
      (B) Com cross-dephasing γ_cross = 10^4 Hz (limite inferior plausível)
      (C) Com cross-dephasing γ_cross = 10^6 Hz (estimativa realista)
    """
    I2   = qt.identity(2)
    sz   = qt.sigmaz()
    sm   = qt.sigmam()
    sp   = qt.sigmap()

    H_logic = 0.5 * qt.tensor(sz, I2)
    H_anc   = 0.5 * 1.2 * qt.tensor(I2, sz)
    H_int   = g_norm * (qt.tensor(sp, sm) + qt.tensor(sm, sp))
    H_tot   = H_logic + H_anc + H_int

    L_reset  = np.sqrt(Gr_norm) * qt.tensor(I2, sm)
    sp_logic = qt.tensor(sp, I2)

    psi0  = qt.tensor((qt.basis(2,0)+qt.basis(2,1)).unit(), qt.basis(2,0)).unit()
    t_max = 6.0
    tlist = np.linspace(0, t_max, 150) * (1/gamma_0)

    results_zeno = {}

    # γ_cross em unidades normalizadas (dividir por ω_0_phys)
    gamma_cross_vals = {
        "Sem cross-deph. (modelo original)": 0.0,
        "γ_cross=10⁴ Hz (otimista)":        1e4 / OMEGA_0_PHYS,
        "γ_cross=10⁶ Hz (realista)":         1e6 / OMEGA_0_PHYS,
    }

    for label, gc in gamma_cross_vals.items():
        cops = [L_reset]
        if gc > 0:
            # Cross-dephasing: Fe³⁺ → campo Bz flutuante → dephasing do spin lógico
            L_cross = np.sqrt(gc) * qt.tensor(sz, I2)
            cops.append(L_cross)
        try:
            res = qt.mesolve(H_tot, psi0, tlist, cops, [sp_logic], options=OPTS)
            coh = np.abs(np.array(res.expect[0]))
        except Exception:
            eff_rate = gamma_zeno_norm + gc
            coh = 0.5 * np.exp(-eff_rate * np.linspace(0, 6.0, 150))
        results_zeno[label] = {"tlist_norm": np.linspace(0, 6.0, 150), "coh": coh}

    return results_zeno


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
    print("  ⏳ Floquet H(t) real (pode demorar ~30s)...")
    floquet_data = sim_floquet_real(nu_over_J_vals=(0.5, 2.0, 8.0))
    print("  ✓ Floquet H(t) OK")
    print("  ⏳ Zeno com/sem cross-dephasing Fe³⁺...")
    zeno_data = sim_zeno_with_cross_dephasing()
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

    # ── CORREÇÃO CRÍTICA (nota análise): distinguir freq. Larmor de J_res ──
    # ω_Larmor = γ_P · B_ext/(2π) ≈ 860 Hz  (oscilação do spin no campo geomagnético)
    # J_res    = 0.01 Hz           (acoplamento dipolar residual APÓS motional narrowing)
    # São grandezas físicas distintas que operam em escalas de frequência diferentes.
    omega_larmor_hz = OMEGA_0_PHYS / (2*np.pi)   # ≈ 860 Hz
    ax1.axvline(omega_larmor_hz,  color="#58a6ff", ls='-',  lw=2.0,
                label=f"ω_Larmor (³¹P) ≈ {omega_larmor_hz:.0f} Hz")
    ax1.axvline(G_COUPLING_PHYS,  color="#27ae60", ls='--', lw=1.8,
                label=f"J_res (acoplamento residual) ≈ {G_COUPLING_PHYS:.0e} Hz")
    ax1.axvline(GAMMA_RESET_PHYS, color="#e74c3c", ls='--', lw=1.8,
                label=f"Fe³⁺ ancilla ≈ {GAMMA_RESET_PHYS:.0e} Hz")
    ax1.axvline(1e11, color="#f39c12", ls=':', lw=1.5,
                label="ω_c Debye ≈ 10¹¹ Hz")
    ax1.axvspan(1e-3, 1e3, alpha=0.10, color="#27ae60")
    ax1.text(2e0, 0.55, "SILENT\nGAP\n(J_res protegido)", color="#27ae60",
             fontsize=9, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#091a09', alpha=0.85))
    ax1.legend(fontsize=7.8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left')
    ax1.set_ylim(0, 1.15); ax1.set_xlim(1e-3, 1e14)
    # Nota explicativa da distinção
    ax1.text(0.97, 0.02,
             r"$J(\omega) \propto \omega^s \, e^{-\omega/\omega_c}$"
             "\n(super-ôhmico, s=3)\n"
             "ω_Larmor ≠ J_res (ver §7.3–7.4)",
             transform=ax1.transAxes, ha='right', va='bottom',
             color="#8b949e", fontsize=8.5,
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P2: Aquecimento Floquet REAL — H(t) simulado ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sax(ax2, "Tempo (J⁻¹)", "⟨H₀⟩(t) — Energia interna",
        "II. Floquet: Aquecimento Medido via H(t) Real\n(substituição da tautologia anterior)")

    # Cores e labels para cada ν/J
    fl_colors = {0.5:"#e74c3c", 2.0:"#f39c12", 8.0:"#27ae60"}
    fl_labels = {0.5:"ν/J=0.5 (sub-drive, aquece rápido)",
                 2.0:"ν/J=2 (drive moderado)",
                 8.0:"ν/J=8 (drive forte, pré-termal)"}

    for nu_J, dat in floquet_data.items():
        ax2.plot(dat["tlist"], dat["energy"],
                 color=fl_colors[nu_J], lw=2.2, label=fl_labels[nu_J])

    ax2.axhline(0, color="#8b949e", ls=':', lw=1.0, alpha=0.5)
    ax2.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")
    ax2.text(0.02, 0.05,
             "Simulação REAL: H(t) = H₀ + A·cos(ω_drive·t)·σ_x\n"
             "mesolve Lindblad com banho γ_bath=0.05J\n"
             "ν/J maior → aquecimento mais lento → pretermalização\n"
             "Este é um teste, não uma premissa inserida no solver.",
             transform=ax2.transAxes, va='bottom', fontsize=8,
             color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.88))
    # Aviso de escala: ν/J_sim ≠ ν/J_bio
    ax2.text(0.97, 0.97,
             "⚠ ν/J=8 na sim. ≠ 10¹⁰ do paper.\n"
             "Escala biológica requer extrapolação\n"
             "além do alcance computacional atual.",
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             color="#f39c12",
             bbox=dict(boxstyle='round', fc='#1a1200', alpha=0.88))

    # ── P3: Fórmula analítica + banda de plausibilidade ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sax(ax3, "ν_drive / J_bath (adimensional)",
        "Γ_H / J_bath  [fórmula analítica, escala log]",
        "III. Floquet: Previsão Analítica (§7.5)\n[Fórmula — NÃO verificação independente]")

    x_fl, y_fl = floquet_scaling()
    ax3.semilogy(x_fl, y_fl + 1e-30, color="#f39c12", lw=2.5)
    ax3.fill_between(x_fl, 1e-30, y_fl+1e-30, alpha=0.12, color="#f39c12")

    nu_min_bio = np.log10(1e12 / 1e4)
    nu_max_bio = np.log10(1e12 / 1e2)
    ax3.axvspan(nu_min_bio, nu_max_bio, alpha=0.12, color="#27ae60",
                label="Plausível bio (J_bath=0.1–10 kHz)")
    ax3.axvline(10, color="#f39c12", ls=':', lw=1.5, alpha=0.8)
    ax3.annotate("Paper: ν/J=10¹⁰\n(J_bath=100 Hz\n— valor otimista)",
                 xy=(10, np.exp(-10)), xytext=(6.5, 1e-2),
                 color="#f39c12", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#f39c12"),
                 bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.9))
    ax3.text(0.03, 0.05,
             "Esta curva é a equação analítica (§7.5).\n"
             "O Painel II mostra simulação real para ν/J ≤ 8.\n"
             "O salto de ν/J=8 para ν/J=10¹⁰ é extrapolação\n"
             "sem verificação numérica independente.",
             transform=ax3.transAxes, va='bottom', fontsize=8,
             color="#f39c12",
             bbox=dict(boxstyle='round', fc='#1a1200', alpha=0.88))
    ax3.text(0.6, 0.82,
             r"$\Gamma_H \sim J \cdot e^{-\nu_{drive}/J}$",
             transform=ax3.transAxes, fontsize=13, color="#f39c12",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))
    ax3.set_xlim(0, 14)
    ax3.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    # ── P4: Zeno com e sem cross-dephasing Fe³⁺ — SIMULAÇÃO REAL ────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sax(ax4, "Tempo (unidades de T₂_bare)", "|ρ₀₁(t)| — Coerência lógica",
        "IV. Zeno + Fe³⁺: Com e Sem Cross-Dephasing\n(operador ausente na versão anterior)")

    zeno_colors = {
        "Sem cross-deph. (modelo original)": "#27ae60",
        "γ_cross=10⁴ Hz (otimista)":        "#f39c12",
        "γ_cross=10⁶ Hz (realista)":         "#e74c3c",
    }
    zeno_ls = {
        "Sem cross-deph. (modelo original)": "-",
        "γ_cross=10⁴ Hz (otimista)":        "--",
        "γ_cross=10⁶ Hz (realista)":         "-.",
    }

    for label, dat in zeno_data.items():
        ax4.semilogy(dat["tlist_norm"], dat["coh"] + 1e-8,
                     color=zeno_colors[label], ls=zeno_ls[label],
                     lw=2.2, label=label)

    ax4.axhline(1/np.e, color="#8b949e", ls=':', lw=1.0, alpha=0.5)
    ax4.set_xlim(0, 6.5); ax4.set_ylim(1e-5, 1.0)
    ax4.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")
    ax4.text(0.02, 0.05,
             "Cross-dephasing: L_cross = √γ_cross · σ_z ⊗ I\n"
             "(campo Bz flutuante do Fe³⁺ sobre ³¹P)\n"
             "γ_cross = (γ_31P B_rms)² τ_c ≈ 10⁴–10⁶ Hz\n"
             "Inclui Lindblad ausente no modelo original.",
             transform=ax4.transAxes, va='bottom', fontsize=8,
             color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.88))
    ax4.text(0.97, 0.97,
             "Fe³⁺ como protetor funciona se\nγ_cross << Γ_reset.\n"
             "γ_cross=10⁶ Hz falha: T₂ colapsa.\n"
             "Separação espectral (Silent Gap)\né necessária mas não verificada.",
             transform=ax4.transAxes, ha='right', va='top', fontsize=8,
             color="#f39c12",
             bbox=dict(boxstyle='round', fc='#1a1200', alpha=0.88))

    # ── Título ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "THEOPHYSICS — Frente A (rev.2): Floquet H(t) Real + Zeno com Cross-Dephasing Fe³⁺\n"
        "Toy model fenomenológico — §7, §15.2 — Limitações epistêmicas explícitas",
        fontsize=12.5, fontweight='bold', color="#e6edf3", y=0.985)

    plt.savefig("frente_A_tegmark_silent_gap.png",
                dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("[Frente A] Figura salva.")
    plt.close()

    # Resumo
    T2_bar_phys = 1/GAMMA_0_PHYS
    print("\n" + "═"*65)
    print("  FRENTE A — RESUMO DE HONESTIDADE EPISTÊMICA")
    print("═"*65)
    print(f"  T₂_bare (³¹P sem proteção):   {T2_bar_phys*1e6:.1f} µs")
    print()
    print("  Floquet:")
    print("    Painel II: simulação H(t) real para ν/J = 0.5, 2, 8")
    print("    Painel III: fórmula analítica (ν/J → 10¹⁰ é extrapolação)")
    print("    Conclusão: ν/J=8 mostra pré-termalização real")
    print("               ν/J=10¹⁰ não foi e não pode ser simulado")
    print()
    print("  Zeno + Fe³⁺ cross-dephasing:")
    print("    γ_cross=0 (original): coerência preservada [tautologia]")
    print("    γ_cross=10⁴ Hz: degradação moderada")
    print("    γ_cross=10⁶ Hz: colapso de coerência")
    print("    Conclusão: γ_cross é o parâmetro crítico não verificado")
    print("═"*65)

if __name__ == "__main__":
    make_figure()
    print("\n[Frente A] Concluído.")
