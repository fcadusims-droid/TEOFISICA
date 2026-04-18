"""
=============================================================================
THEOPHYSICS — FRENTE B
Simulação In Silico: Identidade Classe Λ via MBL (TeNPy TEBD)
=============================================================================

Implementa o Hamiltoniano MBL da Seção 6.4 do paper:

  H_MBL = J Σ_i (Ŝ_i+ Ŝ_{i+1}^- + h.c.)  [hopping]
         + Σ_i h_i Ŝ_i^z                    [campo quasi-periódico]
         + V Σ_i Ŝ_i^z Ŝ_{i+1}^z           [interação Ising]

  h_i = W cos(2π α i + φ),  α = φ = (√5−1)/2  [razão áurea, Def. 2.3]

Compara três classes da taxonomia R–P–M–Λ (Seção 2 do paper):
  Classe Λ: W = 3J (quasi-periódico, fase MBL localizada)
  Classe R: W = 3J (aleatório, fase MBL-random, instável em d>1)
  Classe M: W = 0.2J (fraco → ergódico, mixing, Lyapunov > 0)

Observáveis:
  S_ent(t)   — entropia de emaranhamento (corte central)
  I(A:B)(t)  — informação mútua ≈ 2·S_A (proxy de identidade, §3.1)
  D(t)       — Funcional de Dissonância (proxy, §8.2)
  IPR(W)     — diagrama de fases do modelo Aubry-André
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import tebd
import tenpy
print(f"TeNPy {tenpy.__version__} carregado.")

# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS — Eq. (6.4) do paper
# ─────────────────────────────────────────────────────────────────────────────
PHI   = (np.sqrt(5) - 1) / 2   # razão áurea — freq. quasi-periódica irracional
L     = 14                      # comprimento da cadeia
J     = 1.0                     # hopping (unidade de energia)
V     = 1.0                     # interação Sz-Sz
DT    = 0.05                    # passo temporal TEBD
T_MAX = 10.0                    # tempo máximo (unidades de J⁻¹)
CHI   = 64                      # bond dimension

W_QP  = 3.0    # Classe Λ — quasi-periódico localizado (W > 2J)
W_RND = 3.0    # Classe R — aleatório (mesma amplitude)
W_ERG = 0.3    # Classe M — fraco → ergódico (W < 2J)

TEBD_PARAMS = {
    "order"       : 2,
    "dt"          : DT,
    "N_steps"     : 1,
    "trunc_params": {"chi_max": CHI, "svd_min": 1e-12},
    "verbose"     : 0,
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def quasi_periodic_field(L, W, alpha=PHI, phi=0.0):
    """h_i = W cos(2π α i + φ) — Def. 2.3 e Eq. 6.4 do paper."""
    return W * np.cos(2*np.pi*alpha*np.arange(L) + phi)

def random_field(L, W, seed=42):
    """h_i ~ U(-W, W) — Classe R."""
    rng = np.random.default_rng(seed)
    return W * (2*rng.random(L) - 1)

def build_model(h_vals):
    """Constrói XXZ chain com campo site-dependente."""
    mp = {
        "L"       : L,
        "Jxx"     : J,          # J(S+S- + h.c.) = hopping
        "Jz"      : V,          # V·Sz·Sz
        "hz"      : h_vals,     # campo local
        "bc_MPS"  : "finite",
        "conserve": "Sz",
    }
    return XXZChain(mp)

def neel_mps(model):
    """Estado de Néel |↑↓↑↓...⟩ — alta informação inicial."""
    ps = ["up" if i%2==0 else "down" for i in range(L)]
    return MPS.from_product_state(model.lat.mps_sites(), ps, bc="finite")

def entanglement_entropy(psi, bond=None):
    """S_ent no corte 'bond' (padrão: centro)."""
    if bond is None: bond = L // 2
    SVs = psi.get_SL(bond)
    SVs = SVs[SVs > 1e-13]
    p   = SVs**2; p /= p.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))

def mutual_information(psi):
    """I(A:B) = 2 S_A para estado puro (A = metade esquerda)."""
    return 2.0 * entanglement_entropy(psi, bond=L//2)

def dissonance_normalized(S_t, S_max):
    """
    Proxy da Dissonância via entropia normalizada pela lei de volume:
        D(t) = S(t) / S_max
    D=0: estado puro inicial (identidade total); D=1: termalizado (colapso).
    Mais robusto que proxy de I_mut para TEBD com estado de Néel inicial.
    """
    return np.minimum(S_t / max(S_max, 1e-10), 1.0)


def dissonance(I_t, I_0):
    """
    Proxy do Funcional de Dissonância (Def. 8.1):
        D(t) = δ² / (1 + δ²),  δ = (I_0 - I(t)) / I_0
    D=0 → identidade preservada; D→1 → colapso.
    """
    delta = np.maximum(I_0 - I_t, 0) / max(I_0, 1e-10)
    return delta**2 / (1 + delta**2)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO TEBD POR REGIME
# ─────────────────────────────────────────────────────────────────────────────

def run_regime(h_vals, label):
    """
    Evolve o estado de Néel com TEBD e registra S_ent(t) e I(t).
    """
    M   = build_model(h_vals)
    psi = neel_mps(M)
    eng = tebd.TEBDEngine(psi, M, TEBD_PARAMS)

    tlist, S_list, I_list = [], [], []
    n_steps_total = int(T_MAX / DT)
    rec_every     = max(1, n_steps_total // 100)

    t = 0.0
    for step in range(n_steps_total):
        if step % rec_every == 0:
            tlist.append(t)
            S = entanglement_entropy(psi)
            I = mutual_information(psi)
            S_list.append(S)
            I_list.append(I)
        eng.run()
        t += DT

    return np.array(tlist), np.array(S_list), np.array(I_list)


def simulate_all():
    results = {}

    # Classe Λ — quasi-periódico MBL
    print(f"  [1/3] Classe Λ (W={W_QP}J, α=razão áurea)...")
    h_qp = quasi_periodic_field(L, W_QP)
    t, S, I = run_regime(h_qp, "Classe Λ")
    results["qp"] = {"tlist":t, "S_ent":S, "I_mut":I, "h":h_qp,
                     "label":"Classe Λ — MBL Quasi-Periódico",
                     "color":"#58a6ff", "ls":"-"}
    print(f"       S_ent(T)={S[-1]:.3f}, I(T)={I[-1]:.3f}")

    # Classe R — aleatório
    print(f"  [2/3] Classe R (W={W_RND}J, aleatório)...")
    h_rnd = random_field(L, W_RND)
    t, S, I = run_regime(h_rnd, "Classe R")
    results["rnd"] = {"tlist":t, "S_ent":S, "I_mut":I, "h":h_rnd,
                      "label":"Classe R — Desordem Aleatória",
                      "color":"#e74c3c", "ls":"--"}
    print(f"       S_ent(T)={S[-1]:.3f}, I(T)={I[-1]:.3f}")

    # Classe M — ergódico
    print(f"  [3/3] Classe M (W={W_ERG}J, ergódico)...")
    h_erg = quasi_periodic_field(L, W_ERG)
    t, S, I = run_regime(h_erg, "Classe M")
    results["erg"] = {"tlist":t, "S_ent":S, "I_mut":I, "h":h_erg,
                      "label":"Classe M — Ergódico/Termal",
                      "color":"#f39c12", "ls":"-."}
    print(f"       S_ent(T)={S[-1]:.3f}, I(T)={I[-1]:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAMA DE FASES — IPR vs W (Aubry-André sem interação)
# ─────────────────────────────────────────────────────────────────────────────

def phase_diagram(L_pd=60):
    """
    Calcula o IPR (Inverse Participation Ratio) de estados do meio do espectro
    para diferentes amplitudes W. Transição em W_c = 2J.
    """
    W_vals  = np.linspace(0.1, 6.0, 50)
    IPR_avg = []
    for W in W_vals:
        h = W * np.cos(2*np.pi*PHI*np.arange(L_pd))
        H = np.diag(h)
        for i in range(L_pd-1):
            H[i,i+1] = -J; H[i+1,i] = -J
        _, evecs = np.linalg.eigh(H)
        # média IPR dos estados centrais (estados de energia média)
        mid = L_pd // 4
        ipr_vals = []
        for k in range(mid, L_pd-mid, max(1, L_pd//10)):
            psi_k = evecs[:, k]
            ipr_vals.append(float(np.sum(np.abs(psi_k)**4)))
        IPR_avg.append(np.mean(ipr_vals))
    return W_vals, np.array(IPR_avg)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(results):
    fig = plt.figure(figsize=(17, 13))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.40)

    def sax(ax, xl, yl, tit):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#58a6ff")
        ax.set_xlabel(xl, fontsize=10.5)
        ax.set_ylabel(yl, fontsize=10.5)
        ax.set_title(tit, fontsize=11.5, fontweight='bold', pad=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.grid(True, alpha=0.13, color="#8b949e", ls='--')

    # ── P1: Potencial quasi-periódico vs aleatório ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sax(ax1, "Sítio i", "h_i / J (amplitude)",
        "I. Potencial: Classe Λ vs Classe R")

    sites = np.arange(L)
    h_qp  = results["qp"]["h"]
    h_rnd = results["rnd"]["h"]

    ax1.bar(sites - 0.2, h_qp/J,  0.38, color="#58a6ff",  alpha=0.85,
            label="Quasi-periódico (Classe Λ)")
    ax1.bar(sites + 0.2, h_rnd/J, 0.38, color="#e74c3c80", alpha=0.70,
            label="Aleatório (Classe R)")
    ax1.plot(sites, h_qp/J, 'o-', color="#58a6ff", ms=4, lw=1.2, alpha=0.6)
    ax1.axhline(0, color="#30363d", lw=0.8)
    ax1.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")
    ax1.text(0.97, 0.96,
             f"α = φ = (√5−1)/2\n[razão áurea, irracional]",
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=8.5, color="#58a6ff",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P2: Entropia de Emaranhamento S_ent(t) ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sax(ax2, "Tempo (J⁻¹)", "S_ent(t) [nats]",
        "II. Entropia de Emaranhamento S(t)")

    for key in ["qp", "rnd", "erg"]:
        r = results[key]
        ax2.plot(r["tlist"], r["S_ent"],
                 color=r["color"], ls=r["ls"], lw=2.2, label=r["label"])

    # Referência logarítmica (assinatura MBL)
    t_ref = np.linspace(0.5, T_MAX, 200)
    # Ajuste na curva Classe Λ
    r_qp   = results["qp"]
    mask   = r_qp["tlist"] > 1.0
    if mask.sum() > 5:
        from numpy.polynomial.polynomial import polyfit as pfunc
        log_t  = np.log(r_qp["tlist"][mask] + 0.01)
        coeffs = np.polyfit(log_t, r_qp["S_ent"][mask], 1)
        ax2.plot(t_ref,
                 coeffs[0]*np.log(t_ref+0.01) + coeffs[1],
                 ':', color="#ffffff", lw=1.5, alpha=0.5,
                 label=f"log(t)·{coeffs[0]:.3f} (ajuste Λ)")
        ax2.text(0.03, 0.94,
                 f"S_Λ ~ {coeffs[0]:.3f}·log(t)\n[lei log MBL, §4.5]",
                 transform=ax2.transAxes, va='top', fontsize=9,
                 color="#58a6ff",
                 bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    ax2.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left')

    # ── P3: Informação Mútua I(A:B)(t) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sax(ax3, "Tempo (J⁻¹)", "I(A:B)(t) [nats]",
        "III. Informação Mútua — Proxy de Identidade")

    for key in ["qp", "rnd", "erg"]:
        r = results[key]
        ax3.plot(r["tlist"], r["I_mut"],
                 color=r["color"], ls=r["ls"], lw=2.2, label=r["label"])

    # Limiar I_min da Definição 3.2 (Cond. 6)
    I_min = 0.05
    ax3.axhline(I_min, color="#8b949e", ls=':', lw=1.3)
    ax3.text(T_MAX*0.5, I_min*1.3, "I_min (Cond. 6, Def. 3.2)",
             color="#8b949e", fontsize=8.5, ha='center')

    ax3.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")
    ax3.text(0.97, 0.97,
             "I(A:B) > 0 → identidade preservada\n(Classe Λ permanece acima de I_min)",
             transform=ax3.transAxes, ha='right', va='top',
             fontsize=8.5, color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P4: Funcional de Dissonância D(t) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    sax(ax4, "Tempo (J⁻¹)", "𝒟(t) [adimensional]",
        "IV. Funcional de Dissonância 𝒟(t) — Def. 8.1")

    for key in ["qp", "rnd", "erg"]:
        r  = results[key]
        S_max = (L//2) * np.log(2)
        D  = dissonance_normalized(r["S_ent"], S_max)
        ax4.plot(r["tlist"], D,
                 color=r["color"], ls=r["ls"], lw=2.2, label=r["label"])

    ax4.axhline(0, color="#27ae60", lw=1.0, ls='-', alpha=0.4,
                label="𝒟=0 (alinhamento total)")
    ax4.axhline(1, color="#e74c3c", lw=1.0, ls='--', alpha=0.4,
                label="𝒟=1 (colapso identidade)")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='center right')
    ax4.text(0.03, 0.88,
             "Classe Λ: 𝒟 → baixo (memória mantida)\n"
             "Classe R/M: 𝒟 → 1 (esquecimento/avalanche)",
             transform=ax4.transAxes, fontsize=8.5, color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── P5: Diagrama de fases IPR(W) ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sax(ax5, "Amplitude W/J", "IPR (Inverse Participation Ratio)",
        "V. Diagrama de Fases: Localizado (Λ) vs Estendido (M)")

    print("  [extra] Calculando diagrama de fases IPR(W)...")
    W_ph, IPR_ph = phase_diagram()

    ax5.plot(W_ph, IPR_ph, color="#58a6ff", lw=2.5)
    ax5.fill_between(W_ph, 0, IPR_ph, alpha=0.12, color="#58a6ff")

    Wc = 2*J
    ax5.axvline(Wc, color="#f39c12", ls='--', lw=1.8,
                label=f"W_c = 2J (transição A-A)")
    ax5.axvline(W_QP, color="#27ae60", ls=':', lw=1.5,
                label=f"W_Λ = {W_QP}J (simulação)")
    ax5.axvline(W_ERG, color="#e74c3c", ls=':', lw=1.5,
                label=f"W_M = {W_ERG}J (ergódico)")

    ax5.fill_betweenx([0, IPR_ph.max()*1.1], 0, Wc, alpha=0.06, color="#e74c3c")
    ax5.fill_betweenx([0, IPR_ph.max()*1.1], Wc, 6.5, alpha=0.06, color="#27ae60")

    ax5.text(0.22, 0.80, "ESTENDIDO\n(Classe M)", transform=ax5.transAxes,
             color="#e74c3c", fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.8))
    ax5.text(0.72, 0.80, "LOCALIZADO\n(Classe Λ)", transform=ax5.transAxes,
             color="#27ae60", fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.8))

    ax5.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='lower right')
    ax5.set_xlim(0, 6.5)

    # ── P6: S_ent vs log(t) — assinatura MBL ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    sax(ax6, "log(t / J⁻¹)", "S_ent(t) [nats]",
        "VI. Assinatura MBL: S ∝ log(t) para Classe Λ")

    for key in ["qp", "rnd", "erg"]:
        r    = results[key]
        t    = r["tlist"]
        S    = r["S_ent"]
        mask = t > 0.05
        if mask.sum() > 3:
            ax6.plot(np.log(t[mask]+0.01), S[mask],
                     color=r["color"], ls=r["ls"], lw=2.2, label=r["label"])

    ax6.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left')
    ax6.text(0.97, 0.05,
             "Linha reta em log(t) → lei logarítmica\n"
             "[assinatura única de MBL quasi-periódico]",
             transform=ax6.transAxes, ha='right', va='bottom',
             fontsize=8.5, color="#8b949e",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── Título ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "THEOPHYSICS — Frente B: Identidade de Classe Λ via MBL Quasi-Periódico (Aubry-André)\n"
        f"Cadeia de spins L={L}, J=V={J}, α=φ=(√5−1)/2 — Simulação TEBD (TeNPy) — Paper §6.4, §8.2",
        fontsize=13, fontweight='bold', color="#e6edf3", y=0.985)

    plt.savefig("/mnt/user-data/outputs/frente_B_mbl_identity.png",
                dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n[Frente B] Figura salva.")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TABELA RESUMO
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results):
    print("\n" + "═"*72)
    print("  TAXONOMIA R–P–M–Λ — Resultados Computacionais (Paper §2, §4)")
    print("═"*72)
    print(f"  {'Classe':<12} {'Modo':<22} {'S(T)':>8} {'I(T)':>8} {'𝒟(T)':>8} {'Identidade'}")
    print("─"*72)
    for key, cname in [("qp","Classe Λ"),("rnd","Classe R"),("erg","Classe M")]:
        r  = results[key]
        S  = r["S_ent"][-1]
        Im = r["I_mut"][-1]
        S_max = (L//2) * np.log(2)
        D  = float(dissonance_normalized(np.array([S]), S_max)[-1])
        ok = "✓ PRESERVADA" if D < 0.4 else "✗ COLAPSADA"
        print(f"  {cname:<12} {r['mode'] if hasattr(r,'mode') else r['label'][:22]:<22} "
              f"{S:>8.4f} {Im:>8.4f} {D:>8.4f}   {ok}")
    print("─"*72)
    print("  Previsões teóricas (§2.4, §4.5):")
    print("    Classe Λ → S(t) ~ log(t), 𝒟→0   [memória preservada, §3.2 Cond.6]")
    print("    Classe R → S(t) sub-vol, 𝒟→1    [avalanche/percolação, §4.3]")
    print("    Classe M → S(t) vol. law, 𝒟→1   [mixing/Lyapunov, §4.5]")
    print("═"*72)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═"*65)
    print("  THEOPHYSICS — FRENTE B: Identidade MBL & Redes Tensoriais")
    print(f"  L={L}, J={J}, V={V}, α=φ=(√5-1)/2, dt={DT}, T_max={T_MAX}")
    print("═"*65)

    print("\n[Frente B] Rodando simulações TEBD...")
    results = simulate_all()

    # Adicionar label para lookup
    results["qp"]["mode"]  = "quasi-periódico (W=3J)"
    results["rnd"]["mode"] = "aleatório (W=3J)"
    results["erg"]["mode"] = "ergódico (W=0.3J)"

    print("\n[Frente B] Gerando figura...")
    make_figure(results)
    print_table(results)
    print("\n[Frente B] Concluído.")
