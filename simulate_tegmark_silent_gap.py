import numpy as np
import matplotlib.pyplot as plt
from qutip import brmesolve, tensor, sigmax, sigmay, sigmaz, qeye, basis, mesolve
from scipy.optimize import curve_fit


# --- Model parameters ---
# Dimensionless simulation units are anchored to physical Hz and seconds.
# 1 time unit = 3e-8 s, so 1 frequency unit = 3.33e7 Hz.
time_unit = 3e-8  # seconds per unit
freq_unit = 1 / time_unit  # Hz per unit

# Spin chain parameters
N = 2  # 31P spin chain length
omega_spin = 1e2  # low-frequency spin transition (dimensionless, ~3.3 GHz)
J_zz = 0.05  # weak coupling between adjacent spins

# Bath / phonon environment parameters
alpha = 1e-2  # coupling strength prefactor
omega_cutoff = 1e3  # bath cutoff / high-frequency vibration scale (~3e10 Hz)
omega_vib = 3e3  # dominant protein vibration frequency (~1e11 Hz, protein Debye band)

# Floquet drive parameters
A_drive = 10.0  # drive amplitude in same units as H
omega_drive_list = [0.0, 5e1, 2e2, 5e2, 1e3]  # drive frequencies (dimensionless)

# Floquet drive parameters
A_drive = 10.0  # drive amplitude in same units as H
omega_drive_list = [0.0, 5e1, 2e2, 5e2, 1e3]  # drive frequencies (dimensionless)

# Non-Markovian bath memory time and base decoherence rate
tau_mem = 10.0  # memory time in dimensionless units
gamma_base = 1e-2  # base dephasing strength before spectral suppression

# Time evolution
tmax = 5e1
num_steps = 201
tlist = np.linspace(0.0, tmax, num_steps)

# Operators for 2-spin chain
sigmax_list = [tensor(sigmax() if i == j else qeye(2) for j in range(N)) for i in range(N)]
sigmay_list = [tensor(sigmay() if i == j else qeye(2) for j in range(N)) for i in range(N)]
sigmaz_list = [tensor(sigmaz() if i == j else qeye(2) for j in range(N)) for i in range(N)]

H0 = sum(0.5 * omega_spin * sz for sz in sigmaz_list) + J_zz * sigmaz_list[0] * sigmaz_list[1]

# Initial state: product state in x-basis, maximizing coherence
psi0 = tensor([basis(2, 0) + basis(2, 1) for _ in range(N)])
psi0 = psi0.unit()

# Observable: average transverse coherence
obs = sum(sigmax_list) / N


def spectral_density_highfreq(omega):
    """Super-Ohmic bath model with high-frequency protein vibrations."""
    return alpha * omega**3 * np.exp(-omega / omega_cutoff)


def spectral_density_lowfreq(omega):
    """Low-frequency bath model for a Tegmark-like baseline environment."""
    return alpha * omega * np.exp(-omega / omega_cutoff)


def zeno_suppression(omega_drive):
    """Effective Zeno/Floquet suppression factor from a high-frequency drive."""
    if omega_drive <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (omega_drive / omega_spin) ** 2)


def run_markov(scenario):
    omega_drive = scenario["omega_drive"]
    drive_term = [sum(sigmax_list), lambda t, args: A_drive * np.cos(omega_drive * t)] if scenario["drive"] else None
    H = [H0] if not scenario["drive"] else [H0, drive_term]

    gamma_spin = np.pi * scenario["spectral_density"](omega_spin)
    gamma_eff = gamma_spin * zeno_suppression(omega_drive) * scenario["extra_suppression"]
    collapse_ops = [np.sqrt(max(gamma_eff, 0)) * sz for sz in sigmaz_list]
    solver_options = {"nsteps": 200000, "atol": 1e-8, "rtol": 1e-6}
    sol = mesolve(H, psi0, tlist, collapse_ops, e_ops=[obs], options=solver_options)

    return {
        "label": scenario["label"] + " (Markov)",
        "tlist": tlist,
        "expect": np.real(sol.expect[0]),
        "gamma_eff": gamma_eff,
        "tau_seconds": 1.0 / gamma_eff * time_unit if gamma_eff > 0 else np.nan,
    }


def run_bloch_redfield(scenario):
    omega_drive = scenario["omega_drive"]
    drive_term = [sum(sigmax_list), lambda t, args: A_drive * np.cos(omega_drive * t)] if scenario["drive"] else None
    H = [H0] if not scenario["drive"] else [H0, drive_term]

    # WARNING: standard Bloch-Redfield in QuTiP is based on static frequency components
    # and does not properly incorporate strong time-dependent Floquet drives. The secular
    # approximation breaks down for A cos(omega t) with large omega and strong amplitude.
    if scenario["drive"]:
        print("WARNING: Bloch-Redfield BR estimate is not reliable for strong time-dependent Floquet drives.")
        print("         This result is retained only as an illustrative reference, not as an ab initio proof.")

    a_ops = [(sz, scenario["spectral_density"]) for sz in sigmaz_list]
    solver_options = {"nsteps": 20000, "atol": 1e-8, "rtol": 1e-6}
    sol = brmesolve(H, psi0, tlist, a_ops=a_ops, sec_cutoff=-1, e_ops=[obs], options=solver_options)

    ydata = np.abs(np.real(sol.expect[0]))
    tau_est = np.nan
    if len(ydata) > 0 and ydata[0] > 0:
        y_norm = ydata / ydata[0]
        if y_norm[-1] > 0.95:
            tau_est = tlist[-1] * time_unit
        else:
            target = np.exp(-1)
            below = np.where(y_norm < target)[0]
            if len(below) > 0:
                i = below[0]
                if i == 0:
                    tau_est = tlist[0] * time_unit
                else:
                    tau_est = np.interp(target, y_norm[i - 1 : i + 1][::-1], tlist[i - 1 : i + 1]) * time_unit
            else:
                tau_est = tlist[-1] * time_unit

    return {
        "label": scenario["label"] + " (BR, not reliable for Floquet)",
        "tlist": tlist,
        "expect": np.real(sol.expect[0]),
        "gamma_eff": None,
        "tau_seconds": tau_est,
    }


scenario_defs = [
    {
        "name": "lowfreq_bath",
        "label": "Low-frequency bath",
        "spectral_density": spectral_density_lowfreq,
        "omega_drive": 0.0,
        "drive": False,
        "extra_suppression": 1.0,
    },
    {
        "name": "silent_gap",
        "label": "Protein phonon bath (silent gap)",
        "spectral_density": spectral_density_highfreq,
        "omega_drive": 0.0,
        "drive": False,
        "extra_suppression": 1.0,
    },
    {
        "name": "floquet_zeno",
        "label": "High-frequency Floquet/Zeno protection",
        "spectral_density": spectral_density_highfreq,
        "omega_drive": 1e2,
        "drive": True,
        "extra_suppression": 1.0,
    },
]

results = []
for scenario in scenario_defs:
    results.append(run_markov(scenario))
    results.append(run_bloch_redfield(scenario))


# Plot 1: bath spectral density and silent-gap regime
omega_plot = np.logspace(-3, 4, 501)
J_plot = spectral_density_highfreq(omega_plot)

plt.figure(figsize=(6.5, 4.2))
plt.loglog(omega_plot * freq_unit, J_plot, label=r"$J(\omega)$")
plt.scatter([omega_spin * freq_unit, omega_vib * freq_unit], [spectral_density_highfreq(omega_spin), spectral_density_highfreq(omega_vib)], c=["red", "black"], zorder=5)
plt.annotate("spin transition (~3 GHz)", xy=(omega_spin * freq_unit, spectral_density_highfreq(omega_spin)), xytext=(1e9, 1e-7), arrowprops=dict(arrowstyle="->"))
plt.annotate("protein Debye band (~10^{11} Hz)", xy=(omega_vib * freq_unit, spectral_density_highfreq(omega_vib)), xytext=(1e11, 1e-2), arrowprops=dict(arrowstyle="->"))
plt.xlabel(r"Frequency $\omega$ (Hz)")
plt.ylabel(r"Spectral density $J(\omega)$")
plt.title("Bath spectral density and the Tegmark Silent Gap")
plt.grid(True, which="both", ls="--", alpha=0.45)
plt.legend()
plt.tight_layout()
plt.savefig("silent_gap_spectral_density.png", dpi=200)

# Plot 2: coherence decay comparison
plt.figure(figsize=(6.5, 4.2))
for data in results:
    plt.plot(data["tlist"] * time_unit, np.abs(data["expect"]), label=data["label"])
plt.xlabel("Time (s)")
plt.ylabel(r"Coherence $\langle \sigma_x \rangle$ magnitude")
plt.title("Floquet/Zeno protection of low-frequency 31P coherence")
plt.yscale("linear")
plt.xlim(0, tmax * time_unit)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("coherence_decay_comparison.png", dpi=200)

# Plot 3: estimated coherence time vs protection regime
plt.figure(figsize=(8.0, 4.8))
labels = []
tau_vals = []
for data in results:
    if not np.isnan(data["tau_seconds"]):
        labels.append(data["label"])
        tau_vals.append(data["tau_seconds"])
plt.bar(labels, tau_vals)
plt.yscale("log")
plt.xticks(rotation=30, ha="right")
plt.xlabel("Protection regime")
plt.ylabel(r"Estimated coherence time $\tau$ (s)")
plt.title("Coherence time enhancement: Markovian vs. Bloch-Redfield")
plt.grid(True, axis="y", which="both", ls="--", alpha=0.45)
plt.tight_layout()
plt.savefig("coherence_time_vs_regime.png", dpi=200)

# Print numerical summary
print("\nCoherence time estimates:")
for data in results:
    gamma_label = f"gamma_eff={data['gamma_eff']:.2e}" if data["gamma_eff"] is not None else "gamma_eff=N/A"
    print(f"  {data['label']}: {gamma_label}, tau={data['tau_seconds']:.2e} s")

print("\nNOTE: The collapse of coherence under the standard Bloch-Redfield estimate is visible in these results.")
print("      The current simulation framework does not provide ab initio validation of high-frequency Floquet-Zeno protection.")
print("      A Floquet-Markov master equation or full Floquet averaging is required to settle the 115-day prediction.")

print("\nSaved figures:")
print("  - silent_gap_spectral_density.png")
print("  - coherence_decay_comparison.png")
print("  - coherence_time_vs_regime.png")
