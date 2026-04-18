import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import warnings
warnings.filterwarnings("ignore")

# --- Parameters for the Escatological Pillar ---
# System: Chain of N spins (representing "Book of Life" - structured information)
N = 4  # Small system for computational feasibility
omega = 1.0  # Frequency
gamma = 0.1  # Dissipation rate (thermal death)
T = 2.0  # Temperature (thermal bath)

# Time parameters
tlist = np.linspace(0, 10, 100)

# --- Define the quantum system ---
# Pauli operators for each spin
sx = [tensor([sigmax() if i == j else identity(2) for j in range(N)]) for i in range(N)]
sy = [tensor([sigmay() if i == j else identity(2) for j in range(N)]) for i in range(N)]
sz = [tensor([sigmaz() if i == j else identity(2) for j in range(N)]) for i in range(N)]

# Hamiltonian: Ising-like with nearest neighbor interactions
H = 0
for i in range(N-1):
    H += -omega * sz[i] * sz[i+1]  # ZZ interaction
for i in range(N):
    H += -0.5 * omega * sx[i]  # Transverse field

# Initial state: Product state with some structure (representing "soul" or identity)
psi0 = tensor([basis(2, 0) for _ in range(N)])  # All up state

# --- Define OTOC operators ---
# OTOC: < W(t) V(0) W(t) V(0) > where W and V are local operators
# Choose local operators on first and last spins for scrambling demonstration
W = sx[0]  # Local operator on first spin
V = sx[N-1]  # Local operator on last spin (distant for scrambling)

# --- Thermal bath coupling ---
# Lindblad operators for thermal dissipation
c_ops = []
for i in range(N):
    # Dephasing and relaxation
    c_ops.append(np.sqrt(gamma * (1 + T)) * sz[i])  # Dephasing
    c_ops.append(np.sqrt(gamma * T) * sx[i])  # Relaxation

# NOTE: The recovered OTOC curve below is phenomenological.
# It is a sketch of the IHE/GJW extraction proposal, not the output of a full Floquet
# or open-system Hamiltonian evolution. We keep it for conceptual illustration only.

# --- Function to compute OTOC ---
def compute_otoc(H, psi0, W, V, c_ops, tlist):
    """
    Compute Out-of-Time-Order Correlator: < W(t) V(0) W(t) V(0) >
    This measures quantum scrambling and information loss.
    """
    otoc = []
    for t in tlist:
        # Evolve W forward in time
        U_t = (-1j * H * t).expm()
        W_t = U_t * W * U_t.dag()

        # Evolve V backward in time (from t to 0)
        U_minus_t = (-1j * H * (-t)).expm()
        V_minus_t = U_minus_t * V * U_minus_t.dag()

        # Full OTOC: <psi0| W(t) V(0) W(t) V(0) |psi0>
        # For open systems, we need to use master equation evolution
        # Approximate with unitary evolution for simplicity (thermal effects separate)

        # For dissipative case, compute expectation value
        WVW_V = W_t * V * W_t * V
        exp_val = expect(WVW_V, psi0)
        otoc.append(exp_val)

    return np.array(otoc)

# --- Compute OTOC for different scenarios ---
print("Computing OTOCs for Escatological Analysis...")
print("NOTE: The recovered curve is phenomenological and illustrative, not a direct Schrödinger/master-equation result.")

# Scenario 1: Closed system (no thermal death)
otoc_closed = compute_otoc(H, psi0, W, V, [], tlist)

# Scenario 2: Open system with thermal bath (thermal death)
# For open systems, OTOC computation is more complex.
# Use a physically motivated monotonic decay model for the dissipative case.
# This is still phenomenological, but it avoids artefacts from pure cos^2 oscillations.

def evolve_operator_master_eq(op, H, c_ops, t):
    """Evolve operator under master equation (approximate)"""
    # This is a simplification; proper OTOC in open systems requires more care
    return op  # Placeholder - in reality, need superoperator evolution

gamma_env = gamma * (1 + T)
tau_bath = 1.5
eta = 0.30
otoc_open = np.exp(-gamma_env * tlist) * (1.0 - eta * (1.0 - np.exp(-tlist / tau_bath)))

# --- Information scrambling analysis ---
# Lyapunov exponent from OTOC decay
def fit_lyapunov(tlist, otoc):
    """Fit exponential decay to extract scrambling rate"""
    from scipy.optimize import curve_fit
    def exp_decay(t, A, lambda_L):
        return A * np.exp(-lambda_L * t)
    try:
        popt, _ = curve_fit(exp_decay, tlist, np.abs(otoc), p0=[1, 0.1])
        return popt[1]  # Lyapunov exponent
    except:
        return 0

lambda_closed = fit_lyapunov(tlist, otoc_closed)
lambda_open = fit_lyapunov(tlist, otoc_open)

print(f"Closed-system Lyapunov exponent: {lambda_closed:.3f}")
print(f"Open-system Lyapunov exponent: {lambda_open:.3f}")

# --- Resurrection protocol: Information recovery ---
# Simulate recovery of information from the thermal bath
# Using concept of "holographic encoding" - information preserved in correlations
def resurrection_protocol(otoc_open, tlist, recovery_efficiency=0.7, tau_recov=2.0):
    """
    Simulate information recovery from scrambled state as a gradual extraction process.
    The recovered signal grows over a recovery time and saturates at a fraction of the lost information.
    """
    recovered_otoc = otoc_open + recovery_efficiency * (1.0 - otoc_open) * (1.0 - np.exp(-tlist / tau_recov))
    return recovered_otoc

otoc_recovered = resurrection_protocol(otoc_open, tlist, recovery_efficiency=0.7, tau_recov=2.5)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: OTOC evolution
axes[0,0].plot(tlist, np.abs(otoc_closed), 'b-', label='Closed System (Immortal)', linewidth=2)
axes[0,0].plot(tlist, np.abs(otoc_open), 'r-', label='Open System (Thermal Death)', linewidth=2)
axes[0,0].plot(tlist, np.abs(otoc_recovered), 'g-', label='Phenomenological Resurrection Sketch', linewidth=2)
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('|OTOC|')
axes[0,0].set_title('Out-of-Time-Order Correlator\n(Book of Life Scrambling)')
axes[0,0].legend()
axes[0,0].grid(True)
axes[0,0].set_yscale('log')

# Plot 2: Scrambling rate comparison
scenarios = ['Closed', 'Thermal Death', 'Resurrected']
lambdas = [lambda_closed, lambda_open, fit_lyapunov(tlist, otoc_recovered)]
axes[0,1].bar(scenarios, lambdas, color=['blue', 'red', 'green'])
axes[0,1].set_ylabel('Lyapunov Exponent (Scrambling Rate)')
axes[0,1].set_title('Information Scrambling Rates')
axes[0,1].grid(True, axis='y')

# Plot 3: Recovery gain from thermal death
recovery_gain = np.abs(otoc_recovered) - np.abs(otoc_open)

axes[1,0].plot(tlist, recovery_gain, 'g-', label='Recovery gain vs thermal death', linewidth=2)
axes[1,0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Recovery gain')
axes[1,0].set_title('Recovery gain relative to thermal death\n(Phenomenological resurrection sketch)')
axes[1,0].legend()
axes[1,0].grid(True)

# Plot 4: Phase space representation (conceptual)
# Show how information spreads in the system
entropy_closed = -np.log(np.abs(otoc_closed) + 1e-10)
entropy_open = -np.log(np.abs(otoc_open) + 1e-10)
entropy_recovered = -np.log(np.abs(otoc_recovered) + 1e-10)

axes[1,1].plot(tlist, entropy_closed, 'b-', label='Closed', linewidth=2)
axes[1,1].plot(tlist, entropy_open, 'r-', label='Thermal Death', linewidth=2)
axes[1,1].plot(tlist, entropy_recovered, 'g-', label='Resurrected', linewidth=2)
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Information Entropy')
axes[1,1].set_title('Entropic Dissolution\n(Mathematical Death)')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('escatological_otoc_scrambling.png', dpi=200, bbox_inches='tight')

# --- Quantitative analysis ---
print("\nEscatological Analysis Results:")
print("==============================")
print("1. Identity Dissolution (Thermal Death):")
print(f"   Open-system Lyapunov exponent: {lambda_open:.3f}")
print(f"   Final OTOC: {np.abs(otoc_open[-1]):.3f} (scrambled)")

print("\n2. Information Preservation (Resurrection via IHE/GJW):")
recovery_rate = (np.abs(otoc_recovered[-1]) - np.abs(otoc_open[-1])) / (1 - np.abs(otoc_open[-1]))
print(f"   Recovery fraction (phenomenological sketch): {recovery_rate:.1%}")
print(f"   Final recovered OTOC: {np.abs(otoc_recovered[-1]):.3f}")

print("\n3. Theological Implications:")
print("   - Thermal death dissolves structured identity (soul) into environment")
print("   - Information not erased, but holographically encoded")
print("   - IHE/GJW allows resurrection by decoding environmental correlations")
print("   - This model is a phenomenological sketch of recovery, not a complete proof of perfect identity restoration")
print("   - Closes Section 18: Conceptual proposal for eschatological recovery, with phenomenological illustration")

print("\nSaved figure: escatological_otoc_scrambling.png")
