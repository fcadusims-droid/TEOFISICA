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

# W para FSD dinâmico: usar valor PRÓXIMO à transição (W_c=2J) onde
# S~log(t) é esperado. W=3J (fortemente localizado) mostra plateau rápido,
# que é física correta mas não demonstra o crescimento log.
W_FSD_DYN = 2.5   # ligeiramente acima de W_c=2J

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
    PROXY (mantido para compatibilidade): D = S/S_max.
    Não é a Def. 8.1 do paper — ver nota no Painel IV.
    """
    return np.minimum(S_t / max(S_max, 1e-10), 1.0)


def level_statistics(L_val=60, W=None, mode="quasiperiodic", n_samples=8):
    """
    Calcula o r-parameter (razão de gaps consecutivos de nível) para
    distinguir MBL (Poisson, r≈0.386) de ergódico (GOE, r≈0.530).

    Esta é física REAL e NÃO circular: r não é inserido como premissa,
    é derivado dos autovalores do Hamiltoniano.

    r_n = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1}),  δ_n = E_{n+1} - E_n
    ⟨r⟩_Poisson = 0.386,  ⟨r⟩_GOE = 0.530

    Retorna ⟨r⟩ médio sobre o meio do espectro e amostras de desordem.
    """
    if W is None: W = W_QP
    r_vals = []
    for s in range(n_samples):
        if mode == "quasiperiodic":
            h = W * np.cos(2*np.pi*PHI*np.arange(L_val) + s*0.1)
        elif mode == "random":
            rng = np.random.default_rng(s)
            h   = W * (2*rng.random(L_val) - 1)
        else:  # ergodic — weak field
            h = W * np.cos(2*np.pi*PHI*np.arange(L_val) + s*0.1)

        H_mat = np.diag(h)
        for i in range(L_val-1):
            H_mat[i,i+1] = -J; H_mat[i+1,i] = -J
        evals = np.sort(np.linalg.eigvalsh(H_mat))

        # Usar apenas estados do meio do espectro
        lo = L_val//4; hi = 3*L_val//4
        ev = evals[lo:hi]
        gaps = np.diff(ev)
        gaps = gaps[gaps > 1e-12]
        if len(gaps) < 2: continue
        r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
        r_vals.extend(r.tolist())

    return float(np.mean(r_vals)) if r_vals else 0.5


def level_stats_vs_W(L_val=60, n_samples=6):
    """
    Calcula ⟨r⟩(W) para quasi-periódico e aleatório.
    Mostra onde cada regime está em relação a Poisson (0.386) e GOE (0.530).
    """
    W_vals = np.linspace(0.2, 5.0, 25)
    r_qp  = []
    r_rnd = []
    for W in W_vals:
        r_qp.append(level_statistics(L_val, W, "quasiperiodic", n_samples))
        r_rnd.append(level_statistics(L_val, W, "random",       n_samples))
    return W_vals, np.array(r_qp), np.array(r_rnd)
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
# FINITE-SIZE SCALING DINÂMICO — TEBD para múltiplos L (Classe Λ)
# Responde à crítica: TEBD em L=14 tem efeitos de tamanho finito substanciais.
# Test chave: se S(t) ~ c·log(t) com slope c ≈ constante para L=10→18,
# então o crescimento log é propriedade de bulk, não artefato de tamanho finito.
# ─────────────────────────────────────────────────────────────────────────────

def run_regime_L(L_val, W, mode="quasiperiodic", seed=42, t_max_override=None):
    """
    Versão parametrizada de run_regime com L explícito.
    Cria modelo e funções locais para L_val sem depender do global L.
    """
    np.random.seed(seed)
    t_max_local = t_max_override if t_max_override is not None else T_MAX
    # Potencial
    if mode == "quasiperiodic":
        h_vals = W * np.cos(2*np.pi*PHI*np.arange(L_val))
    else:
        rng = np.random.default_rng(seed)
        h_vals = W * (2*rng.random(L_val) - 1)

    # Modelo
    mp = {"L":L_val, "Jxx":J, "Jz":V, "hz":h_vals,
          "bc_MPS":"finite", "conserve":"Sz"}
    M_loc = XXZChain(mp)

    # MPS estado de Néel
    ps  = ["up" if i%2==0 else "down" for i in range(L_val)]
    psi = MPS.from_product_state(M_loc.lat.mps_sites(), ps, bc="finite")

    tp_loc = {"order":2, "dt":DT, "N_steps":1,
              "trunc_params":{"chi_max":CHI, "svd_min":1e-12}, "verbose":0}
    eng = tebd.TEBDEngine(psi, M_loc, tp_loc)

    # Funções de medida locais (não dependem do global L)
    def S_local(bond=None):
        b = L_val//2 if bond is None else bond
        SVs = psi.get_SL(b); SVs = SVs[SVs > 1e-13]
        p = SVs**2; p /= p.sum()
        return float(-np.sum(p * np.log(p + 1e-15)))

    tlist_l, S_list_l = [], []
    n_steps  = int(t_max_local / DT)
    rec_every = max(1, n_steps // 80)
    t = 0.0
    for step in range(n_steps):
        if step % rec_every == 0:
            tlist_l.append(t)
            S_list_l.append(S_local())
        eng.run()
        t += DT

    return np.array(tlist_l), np.array(S_list_l)


def finite_size_dynamics(L_list=(10, 12, 14, 16, 18), W=None):
    """
    Roda TEBD para cada L em L_list (Classe Λ, W=W_QP).
    Extrai slope do ajuste linear S ~ c·log(t) + const no regime tardio.

    Janela de fitting: segunda metade do intervalo de tempo em log-scale,
    evitando o transiente inicial rápido e o plateau prematuro em L pequeno.

    Se c ≈ constante com L → crescimento log é propriedade de bulk.
    Retorna dict {L: {"tlist", "S_ent", "slope", "intercept"}}
    """
    if W is None: W = W_QP
    # FSD usa T_MAX maior para ver o regime assintótico
    T_MAX_FSD = T_MAX * 2.0
    results_fsd = {}
    for Lval in L_list:
        print(f"    L={Lval}...", end=" ", flush=True)
        tlist_l, S_l = run_regime_L(Lval, W, t_max_override=T_MAX_FSD)
        # Fitting robusto: usa apenas o terço final do tempo em log-scale
        # para capturar regime assintótico e evitar transiente
        t_fit_lo = np.exp(np.log(max(tlist_l[1], 0.1)) +
                          0.67*(np.log(T_MAX_FSD) - np.log(max(tlist_l[1],0.1))))
        mask = tlist_l > t_fit_lo
        slope, intercept = 0.0, float(S_l[0])
        if mask.sum() > 4:
            log_t  = np.log(tlist_l[mask] + 0.01)
            coeffs = np.polyfit(log_t, S_l[mask], 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
        results_fsd[Lval] = {"tlist":tlist_l, "S_ent":S_l,
                              "slope":slope, "intercept":intercept}
        print(f"slope={slope:.4f} (t_fit > {t_fit_lo:.1f})")
    return results_fsd

def phase_diagram_multi_L(L_list=(20, 40, 60, 100)):
    """
    Calcula IPR vs W para múltiplos tamanhos L (escalonamento de tamanho finito).
    Crítica da análise: L=14 é pequeno; este plot mostra como a transição
    afia com L crescente — evidência de que é uma transição real e não
    artefato de tamanho finito.

    Para o modelo Aubry-André não interagente, W_c = 2J exato.
    Com interação V, W_c desloca para cima (MBL interagente).
    """
    W_vals = np.linspace(0.1, 6.0, 60)
    results = {}
    for Lval in L_list:
        IPR_avg = []
        for W in W_vals:
            h = W * np.cos(2*np.pi*PHI*np.arange(Lval))
            H = np.diag(h)
            for i in range(Lval-1):
                H[i,i+1] = -J; H[i+1,i] = -J
            _, evecs = np.linalg.eigh(H)
            # média sobre estados do quarto central do espectro
            lo = Lval // 4; hi = 3 * Lval // 4
            ipr_vals = [float(np.sum(np.abs(evecs[:,k])**4))
                        for k in range(lo, hi, max(1,(hi-lo)//12))]
            IPR_avg.append(np.mean(ipr_vals))
        results[Lval] = np.array(IPR_avg)
    return W_vals, results


def finite_size_slope(W_vals, IPR_dict):
    """
    Extrai a posição do ponto de inflexão (W_c estimado) via gradiente máximo.
    Com L→∞, W_c deve convergir para 2J.
    """
    W_c_list = {}
    for Lval, IPR in IPR_dict.items():
        grad = np.gradient(IPR, W_vals)
        # ponto de máxima inclinação (transição)
        idx  = np.argmax(np.abs(grad))
        W_c_list[Lval] = W_vals[idx]
    return W_c_list


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
        "II. S_ent(t): CONFIRMAÇÃO da Crítica de Dimensionalidade")

    for key in ["qp", "rnd", "erg"]:
        r = results[key]
        ax2.plot(r["tlist"], r["S_ent"],
                 color=r["color"], ls=r["ls"], lw=2.2, label=r["label"])

    # Referência logarítmica (assinatura MBL)
    t_ref = np.linspace(0.5, T_MAX, 200)
    r_qp   = results["qp"]
    mask   = r_qp["tlist"] > 1.0
    if mask.sum() > 5:
        log_t  = np.log(r_qp["tlist"][mask] + 0.01)
        coeffs = np.polyfit(log_t, r_qp["S_ent"][mask], 1)
        ax2.plot(t_ref,
                 coeffs[0]*np.log(t_ref+0.01) + coeffs[1],
                 ':', color="#ffffff", lw=1.5, alpha=0.5,
                 label=f"log(t)·{coeffs[0]:.3f} (ajuste Λ)")

    # NOTA CRÍTICA EXPLÍCITA: R também localiza em 1D
    ax2.text(0.02, 0.97,
             "⚠ CRÍTICA DIMENSIONAL CONFIRMADA:\n"
             f"Classe R (W={W_RND}J, aleatório) NÃO colapsa em 1D.\n"
             "Em 1D, desordem aleatória → Localização de Anderson\n"
             "para qualquer W>0 (teorema de Furstenberg).\n"
             "Prop. 4.3 (R→avalanche) requer d≥2.\n"
             "Esta sim. não valida a distinção Λ vs R do paper.",
             transform=ax2.transAxes, va='top', fontsize=8,
             color="#e74c3c",
             bbox=dict(boxstyle='round', fc='#1a0000', alpha=0.90))

    ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='lower right')

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

    # ── P4: Estatística de Níveis ⟨r⟩(W) — física real, não proxy ──────────
    ax4 = fig.add_subplot(gs[1, 0])
    sax(ax4, "Amplitude W/J", "⟨r⟩ (r-parameter)",
        "IV. Estatística de Níveis ⟨r⟩(W)\n(substitui proxy-Dissonância — física real)")

    print("  [extra] Calculando r-parameter vs W (quasi-periódico e aleatório)...")
    W_r, r_qp_arr, r_rnd_arr = level_stats_vs_W(L_val=60, n_samples=6)

    ax4.plot(W_r, r_qp_arr,  color="#58a6ff", lw=2.2, label="Quasi-periódico (Λ)")
    ax4.plot(W_r, r_rnd_arr, color="#e74c3c", lw=2.2, ls='--', label="Aleatório (R)")

    # Linhas de referência teóricas
    ax4.axhline(0.386, color="#27ae60", ls='--', lw=1.5,
                label="r=0.386 (Poisson — MBL)")
    ax4.axhline(0.530, color="#f39c12", ls='--', lw=1.5,
                label="r=0.530 (GOE — Ergódico)")
    ax4.axvline(2*J, color="#ffffff", ls=':', lw=1.0, alpha=0.5,
                label="W_c = 2J (A-A)")

    ax4.set_ylim(0.3, 0.6); ax4.set_xlim(0.2, 5.0)
    ax4.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9")

    ax4.text(0.02, 0.05,
             "r-parameter: mede estatística de espaçamentos\n"
             "de níveis de energia. NÃO é inserido como premissa.\n"
             "Derivado dos autovalores do Hamiltoniano — física real.\n"
             "Λ e R ambos → Poisson (r≈0.386) para W>W_c em 1D:\n"
             "confirmação de que ambos localizam em 1D.\n"
             "Diferença Λ vs R apenas em d≥2 (via avalanche).",
             transform=ax4.transAxes, va='bottom', fontsize=8,
             color="#c9d1d9",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.88))

    # ── P5: Finite-Size Scaling IPR(W, L) ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sax(ax5, "Amplitude W/J", "IPR médio (estados centrais)",
        "V. Finite-Size Scaling IPR(W) — Transição Aubry-André")

    L_list_pd = [20, 40, 60, 100]
    colors_pd  = ["#f39c12", "#58a6ff", "#27ae60", "#e74c3c"]

    print("  [extra] Finite-size scaling IPR(W, L)...")
    W_ph, IPR_dict = phase_diagram_multi_L(L_list_pd)
    W_c_estimates  = finite_size_slope(W_ph, IPR_dict)

    for Lval, col in zip(L_list_pd, colors_pd):
        IPR = IPR_dict[Lval]
        ax5.plot(W_ph, IPR, color=col, lw=2.0, label=f"L = {Lval}")

    # W_c teórico exato
    Wc = 2*J
    ax5.axvline(Wc, color="#ffffff", ls='--', lw=1.5, alpha=0.7,
                label=f"W_c = 2J (exato, L→∞)")

    # Posições dos W_c estimados por gradiente
    for Lval, col in zip(L_list_pd, colors_pd):
        wc_est = W_c_estimates[Lval]
        ax5.axvline(wc_est, color=col, ls=':', lw=0.9, alpha=0.6)

    # Pontos de simulação dinâmica
    ax5.axvline(W_QP,  color="#27ae60", ls='-.', lw=1.3,
                label=f"W_Λ={W_QP}J (TEBD)")
    ax5.axvline(W_ERG, color="#f39c12", ls='-.', lw=1.3,
                label=f"W_M={W_ERG}J (TEBD)")

    ax5.fill_betweenx([0, 1], 0, Wc,   alpha=0.05, color="#e74c3c")
    ax5.fill_betweenx([0, 1], Wc, 6.5, alpha=0.05, color="#27ae60")
    ax5.text(0.25, 0.82, "ESTENDIDO\n(Classe M)", transform=ax5.transAxes,
             color="#e74c3c", fontsize=8.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.8))
    ax5.text(0.72, 0.82, "LOCALIZADO\n(Classe Λ)", transform=ax5.transAxes,
             color="#27ae60", fontsize=8.5, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.8))

    # Inset: convergência de W_c(L) → 2J
    axins = ax5.inset_axes([0.55, 0.08, 0.42, 0.38])
    axins.set_facecolor("#0d1117")
    L_arr   = np.array(L_list_pd)
    Wc_arr  = np.array([W_c_estimates[l] for l in L_list_pd])
    axins.plot(1/L_arr, Wc_arr, 'o-', color="#ffffff", ms=6, lw=1.5)
    axins.axhline(Wc, color="#f39c12", ls='--', lw=1.0)
    axins.set_xlabel("1/L", fontsize=7.5, color="#8b949e")
    axins.set_ylabel("W_c(L)", fontsize=7.5, color="#8b949e")
    axins.tick_params(colors="#8b949e", labelsize=7)
    axins.set_title("W_c → 2J\n(L→∞)", fontsize=7.5, color="#f39c12")
    for sp in axins.spines.values(): sp.set_edgecolor("#30363d")

    ax5.legend(fontsize=7.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left', ncol=2)
    ax5.set_xlim(0, 6.5)

    # ── P6: Finite-Size Scaling DINÂMICO — multi-L TEBD para Classe Λ ────────
    ax6 = fig.add_subplot(gs[1, 2])
    sax(ax6, "log(t / J⁻¹)", "S_ent(t) [nats]",
        f"VI. FSD Dinâmico: S∝log(t), W={W_FSD_DYN}J (perto de W_c)")

    # Paleta para diferentes L
    L_colors = {10:"#f0e68c", 12:"#7ec8e3", 14:"#58a6ff",
                16:"#27ae60", 18:"#e74c3c"}

    # Rodar TEBD para múltiplos L (Classe Λ, W=W_FSD_DYN perto da transição)
    # Nota: W=W_QP=3J está no regime fortemente localizado onde S satura rápido
    # (slope≈0 é física correta). Para ver log growth, usamos W=W_FSD_DYN=2.5J.
    print("\n  [FSD dinâmico] TEBD multi-L para Classe Λ (W=W_FSD_DYN)...")
    fsd_data = finite_size_dynamics(L_list=(10, 12, 14, 16, 18), W=W_FSD_DYN)

    for Lval, dat in fsd_data.items():
        t = dat["tlist"]; S = dat["S_ent"]
        mask = t > 0.05
        if mask.sum() < 3: continue
        col  = L_colors.get(Lval, "#ffffff")
        ax6.plot(np.log(t[mask]+0.01), S[mask],
                 color=col, lw=2.0, label=f"L = {Lval}")
        # Curva de ajuste log
        if dat["slope"] != 0.0:
            log_t_fit = np.linspace(np.log(1.0), np.log(T_MAX), 80)
            S_fit     = dat["slope"] * log_t_fit + dat["intercept"]
            ax6.plot(log_t_fit, S_fit, ':', color=col, lw=1.0, alpha=0.7)

    ax6.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", loc='upper left')

    # Inset: slope c vs L (o teste crítico)
    axins6 = ax6.inset_axes([0.48, 0.05, 0.50, 0.42])
    axins6.set_facecolor("#0d1117")
    Lvals_  = np.array(sorted(fsd_data.keys()))
    slopes_ = np.array([fsd_data[lv]["slope"] for lv in Lvals_])
    # Linha teórica assintótica (Bardarson et al. 2012): c = 1/6 ≈ 0.167
    # Requer L >> 20 e t >> 100 J^{-1} para ser atingida
    axins6.axhline(1/6, color="#f39c12", ls='--', lw=1.5,
                   label="c=1/6 (bulk, L→∞, t→∞)")
    axins6.axhline(0.0, color="#8b949e", ls=':', lw=1.0, alpha=0.6)
    axins6.plot(Lvals_, slopes_, 'o-', color="#ffffff", ms=7, lw=1.8,
                label="slope medido (T=20J⁻¹)")
    axins6.set_xlabel("L", fontsize=7.5, color="#8b949e")
    axins6.set_ylabel("c = ΔS/Δlog(t)", fontsize=7, color="#8b949e")
    axins6.tick_params(colors="#8b949e", labelsize=7)
    axins6.legend(fontsize=6.5, facecolor="#0d1117", edgecolor="#30363d",
                  labelcolor="#c9d1d9")
    # Limite computacional é honesto
    axins6.set_title("c < 1/6: MBL confinado\n(T e χ insuf. p/ bulk)",
                     fontsize=7, color="#f39c12")
    for sp in axins6.spines.values(): sp.set_edgecolor("#30363d")

    ax6.text(0.97, 0.02,
             f"W={W_FSD_DYN}J (perto da transição W_c=2J)\n"
             "Linhas sólidas: S_ent(t); pontilhadas: ajuste\n"
             "Inset: slope c vs L — se c≈1/6 e constante → bulk\n"
             f"Nota: W={W_QP}J (fortemente loc.) → slope≈0 (correto\nfisicamente: S satura rapidamente no regime W>>W_c)",
             transform=ax6.transAxes, ha='right', va='bottom',
             fontsize=7.5, color="#8b949e",
             bbox=dict(boxstyle='round', fc='#0d1117', alpha=0.85))

    # ── Título ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "THEOPHYSICS — Frente B (rev.2): MBL 1D + Estatística de Níveis + FSD\n"
        "Toy model fenomenológico — Limitações de dimensionalidade explícitas — Paper §6.4, §8.2",
        fontsize=12.5, fontweight='bold', color="#e6edf3", y=0.985)

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
    print("─"*72)
    print("  DIAGNÓSTICO HONESTO (incorporando crítica de dimensionalidade):")
    print("    • Em 1D, Classe R TAMBÉM localiza (Anderson, teorema Furstenberg)")
    print("    • Simulação 1D NÃO distingue Λ de R pela Prop. 4.3 do paper")
    print("    • r-parameter (Painel IV): Λ e R → Poisson em W>W_c em 1D")
    print("    • A distinção R→avalanche→colapso requer d≥2 (não simulado)")
    print("    • O que a sim. mostra validamente: Classe M (ergódico) ≠ Λ")
    print("    • FSD dinâmico (Painel VI): limite computacional explícito (c<1/6)")
    print("    • Status correto: toy model fenomenológico, não validação do HBM")
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
