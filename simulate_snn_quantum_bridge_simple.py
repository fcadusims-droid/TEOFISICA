import numpy as np
import matplotlib.pyplot as plt

# --- Parameters from previous quantum simulation ---
# These coherence times are assumed/hypothetical and conditional on the Floquet-Zeno theory.
# They are not numerically validated by a full Bloch-Redfield/Floquet-Markov treatment yet.
coherence_times = {
    "lowfreq_bath": 1e-4,  # Low-frequency bath (short coherence)
    "silent_gap": 1.0,     # Protein phonon bath (long coherence)
    "floquet_zeno": 1e7,   # High-frequency Floquet/Zeno (very long coherence)
}

# --- Neural Network Parameters ---
N_neurons = 100  # Number of neurons in the network
duration = 20.0  # Simulation duration in seconds (longer run for more avalanche samples)
dt = 0.001  # Time step in seconds

# Leaky Integrate-and-Fire neuron parameters
tau_m = 0.020  # Membrane time constant (s)
tau_ref = 0.001  # Refractory period (s)
V_th = -53e-3  # Threshold (V)
V_reset = -65e-3  # Reset potential (V)
V_rest = -70e-3  # Resting potential (V) - more negative than threshold
I_bias = 0.90  # Bias current (A, but scaled) - set to produce observable network activity

# Synaptic parameters
tau_syn = 0.010  # Synaptic time constant (s)
w_exc = 0.005  # Excitatory weight (V)
w_inh = -0.005  # Inhibitory weight (V)
p_conn = 0.05  # Connection probability

# External input
rate_ext = 20  # External Poisson input rate (Hz)
w_ext = 0.01  # External input weight (V)

# --- Quantum influence function ---
def quantum_modulation(coherence_tau, t):
    """
    Modulate neural firing based on quantum coherence time.
    Longer coherence -> less noise, more stable firing.
    Shorter coherence -> more noise, more variable firing.
    """
    base_noise = 0.02
    coherence_factor = 1.0 / (1.0 + coherence_tau / 1e-2)
    noise_level = base_noise * coherence_factor
    return noise_level

# --- Simulation function ---
def simulate_snn_with_quantum_influence(scenario_name, coherence_tau):
    print(f"Simulating SNN with {scenario_name} (tau = {coherence_tau:.2e} s)")

    # Time array
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Neuron state variables
    v = np.full(N_neurons, V_rest)  # Membrane potential
    I_syn = np.zeros(N_neurons)  # Synaptic current
    last_spike = np.full(N_neurons, -tau_ref)  # Last spike time

    # Spike times
    spike_times = []
    spike_neurons = []

    # Synaptic connections (random)
    exc_conn = np.random.rand(N_neurons, N_neurons) < p_conn
    inh_conn = np.random.rand(N_neurons, N_neurons) < p_conn
    np.fill_diagonal(exc_conn, False)  # No self-connections
    np.fill_diagonal(inh_conn, False)

    for i in range(n_steps):
        t_current = t[i]

        # Refractory check
        refractory = (t_current - last_spike) < tau_ref
        v[refractory] = V_reset

        # Quantum influence: add noise based on coherence
        noise_level = quantum_modulation(coherence_tau, t_current)
        quantum_noise = noise_level * np.random.randn(N_neurons)

        # External input (Poisson)
        ext_spikes = np.random.rand(N_neurons) < (rate_ext * dt)
        I_ext = ext_spikes * w_ext

        # Synaptic input from network
        I_syn_input = np.zeros(N_neurons)
        if len(spike_times) > 0:
            recent_spikes = np.array(spike_times[-10:])  # Last 10 spikes
            recent_neurons = np.array(spike_neurons[-10:])
            for j, (spike_t, neuron_idx) in enumerate(zip(recent_spikes, recent_neurons)):
                if t_current - spike_t < 0.01:  # Within 10ms
                    decay = np.exp(-(t_current - spike_t) / tau_syn)
                    I_syn_input += exc_conn[:, neuron_idx] * w_exc * decay
                    I_syn_input += inh_conn[:, neuron_idx] * w_inh * decay

        # Update synaptic current
        I_syn += dt * (-I_syn / tau_syn) + I_syn_input * dt

        # Update membrane potential
        dv = dt * ((V_rest - v) / tau_m + I_syn + I_bias + I_ext + quantum_noise)
        v += dv

        # Check for spikes
        spiked = (v >= V_th) & ~refractory
        if np.any(spiked):
            spike_times.extend([t_current] * np.sum(spiked))
            spike_neurons.extend(np.where(spiked)[0])
            v[spiked] = V_reset
            last_spike[spiked] = t_current

    return np.array(spike_times), np.array(spike_neurons), t

def compute_avalanches(spike_times, duration, bin_size=0.02):
    """Compute avalanche sizes, durations, and effective branching ratio."""
    bins = np.arange(0, duration + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=bins)
    active = counts > 0

    avalanches = []
    durations = []
    sigma_events = []
    start = None

    for i, active_bin in enumerate(active):
        if active_bin and start is None:
            start = i
        elif not active_bin and start is not None:
            avalanche = counts[start:i]
            if avalanche.sum() > 0:
                avalanches.append(avalanche.sum())
                durations.append((i - start) * bin_size)
                for j in range(len(avalanche) - 1):
                    if avalanche[j] > 0:
                        sigma_events.append(avalanche[j + 1] / avalanche[j])
            start = None

    if start is not None:
        avalanche = counts[start:]
        if avalanche.sum() > 0:
            avalanches.append(avalanche.sum())
            durations.append((len(active) - start) * bin_size)
            for j in range(len(avalanche) - 1):
                if avalanche[j] > 0:
                    sigma_events.append(avalanche[j + 1] / avalanche[j])

    sigma = np.nan if len(sigma_events) == 0 else np.mean(sigma_events)
    return counts, avalanches, durations, sigma

# --- Empirical reference from neuronal avalanche literature ---
# Beggs & Plenz (2003) report size exponent ~3/2 and duration exponent ~2.0.
# Shriki et al. (2013) report similar critical scaling in large-scale human MEG recordings.

def fit_power_law_histogram(values, n_bins=8, x_min=None, x_max=None):
    if len(values) < 10:
        return np.nan, np.nan, np.nan
    if x_min is None:
        x_min = max(1, np.min(values))
    if x_max is None:
        x_max = np.max(values)
    bins = np.logspace(np.log10(x_min), np.log10(x_max + 1e-6), n_bins)
    counts, edges = np.histogram(values, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = (counts > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(np.log(centers[mask]), np.log(counts[mask]), 1)
    alpha = -slope
    return alpha, np.exp(intercept), centers[mask]

# --- Run simulations for each quantum scenario ---
results = {}
for scenario, tau in coherence_times.items():
    spike_times, spike_neurons, t = simulate_snn_with_quantum_influence(scenario, tau)
    counts, avalanches, durations, sigma = compute_avalanches(spike_times, duration)
    results[scenario] = {
        'spike_times': spike_times,
        'spike_neurons': spike_neurons,
        't': t,
        'tau': tau,
        'counts': counts,
        'avalanches': avalanches,
        'durations': durations,
        'sigma': sigma,
    }

# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Raster plots (reference only)
for i, (scenario, data) in enumerate(results.items()):
    ax = axes[i]
    if len(data['spike_times']) > 0:
        ax.scatter(data['spike_times'], data['spike_neurons'], s=1, c='k')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'{scenario} (τ = {data["tau"]:.1e} s)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, N_neurons)
    if i == 2:
        ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('snn_quantum_raster_reference.png', dpi=200, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 2: Avalanche size distribution
ax = axes[0]
for scenario, data in results.items():
    if len(data['avalanches']) > 0:
        counts, bins = np.histogram(data['avalanches'], bins=np.logspace(-1, np.log10(max(data['avalanches']) + 1), 20))
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        ax.loglog(bin_centers, counts + 1, marker='o', label=f'{scenario}')
# Empirical literature reference for avalanche sizes
ref_x = np.logspace(-1, 2, 50)
ref_y = ref_x ** (-1.5)
ref_y *= 1e3 / np.max(ref_y)
ax.loglog(ref_x, ref_y, 'k--', label='Literature ref: size ∼ S^-1.5')
ax.set_xlabel('Avalanche size')
ax.set_ylabel('Count')
ax.set_title('Avalanche size distribution (log-log)')
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

# Plot 3: Avalanche duration distribution
ax = axes[1]
for scenario, data in results.items():
    if len(data['durations']) > 0:
        counts, bins = np.histogram(data['durations'], bins=np.logspace(-3, np.log10(max(data['durations']) + 1e-6), 20))
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        ax.loglog(bin_centers, counts + 1, marker='o', label=f'{scenario}')
# Empirical literature reference for avalanche durations
ref_t = np.logspace(-3, 0.5, 50)
ref_counts = ref_t ** (-2.0)
ref_counts *= 1e3 / np.max(ref_counts)
ax.loglog(ref_t, ref_counts, 'k--', label='Literature ref: duration ∼ T^-2.0')
ax.set_xlabel('Avalanche duration (s)')
ax.set_ylabel('Count')
ax.set_title('Avalanche duration distribution (log-log)')
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('snn_quantum_avalanche_analysis.png', dpi=200, bbox_inches='tight')

def characterize_avalanche_distribution(avalanches):
    if len(avalanches) == 0:
        return "no avalanches"
    sizes = np.array(avalanches)
    bins = np.logspace(np.log10(max(0.9, sizes.min())), np.log10(sizes.max() + 1), 10)
    counts, _ = np.histogram(sizes, bins=bins)
    if np.sum(counts) == 0:
        return "no avalanches"
    relative = counts / np.sum(counts)
    mode = np.argmax(relative)
    if relative[mode] > 0.40:
        mode_size = np.sqrt(bins[mode] * bins[mode + 1])
        return f"mode-dominated at size ~{mode_size:.1f}"
    return "multi-scale"

# --- Quantitative analysis ---
print("\nAvalanche-focused analysis:")
for scenario, data in results.items():
    total_spikes = len(data['spike_times'])
    n_avalanches = len(data['avalanches'])
    mean_size = np.mean(data['avalanches']) if n_avalanches > 0 else 0
    mean_duration = np.mean(data['durations']) if n_avalanches > 0 else 0
    sigma = data['sigma']
    mode_comment = characterize_avalanche_distribution(data['avalanches'])
    size_alpha, _, _ = fit_power_law_histogram(data['avalanches'], n_bins=8)
    duration_alpha, _, _ = fit_power_law_histogram(data['durations'], n_bins=8)
    size_note = "" if not np.isnan(size_alpha) else " (cannot fit exponent)"
    duration_note = "" if not np.isnan(duration_alpha) else " (cannot fit exponent)"
    sample_note = "" if n_avalanches >= 20 else " (insufficient avalanche samples for robust scaling)"
    print(f"{scenario}: spikes={total_spikes}, avalanches={n_avalanches}, mean size={mean_size:.1f}, mean duration={mean_duration:.3f}s, sigma={sigma:.2f}, size exp={size_alpha:.2f}{size_note}, duration exp={duration_alpha:.2f}{duration_note}, distribution={mode_comment}{sample_note}")

print("\nReference empirical exponents: avalanche size ~1.5 (Beggs & Plenz 2003), avalanche duration ~2.0 (Shriki et al. 2013).")
print("      The plotted literature reference curves are approximate benchmarks, not raw dataset overlays.")
print("      Low-count regimes or mode-dominated distributions do not support a power-law criticality claim.")
print("Saved figure: snn_quantum_avalanche_analysis.png")