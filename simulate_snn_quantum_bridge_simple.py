import numpy as np
import matplotlib.pyplot as plt

# --- Parameters from previous quantum simulation ---
# Coherence times from the Bloch-Redfield simulation (in seconds) - scaled for effect
coherence_times = {
    "lowfreq_bath": 1e-4,  # Low-frequency bath (short coherence)
    "silent_gap": 1.0,     # Protein phonon bath (long coherence)
    "floquet_zeno": 1e7,   # High-frequency Floquet/Zeno (very long coherence)
}

# --- Neural Network Parameters ---
N_neurons = 100  # Number of neurons in the network
duration = 1.0  # Simulation duration in seconds
dt = 0.001  # Time step in seconds

# Leaky Integrate-and-Fire neuron parameters
tau_m = 0.020  # Membrane time constant (s)
tau_ref = 0.0  # Refractory period (s) - disabled for now
V_th = -50e-3  # Threshold (V)
V_reset = -65e-3  # Reset potential (V)
V_rest = -70e-3  # Resting potential (V) - more negative than threshold
I_bias = 1000e-3  # Bias current (A, but scaled) - back to working value

# Synaptic parameters
tau_syn = 0.005  # Synaptic time constant (s)
w_exc = 0.01  # Excitatory weight (V)
w_inh = -0.01  # Inhibitory weight (V)
p_conn = 0.1  # Connection probability

# External input
rate_ext = 200  # External Poisson input rate (Hz)
w_ext = 0.01  # External input weight (V)

# --- Quantum influence function ---
def quantum_modulation(coherence_tau, t):
    """
    Modulate neural firing based on quantum coherence time.
    Longer coherence -> less noise, more stable firing.
    Shorter coherence -> more noise, more variable firing.
    """
    # Base noise level inversely proportional to coherence time
    base_noise = 0.01  # Increased
    coherence_factor = np.clip(1.0 / (coherence_tau * 1e4), 0.1, 100.0)  # Scale factor adjusted
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

# --- Run simulations for each quantum scenario ---
results = {}
for scenario, tau in coherence_times.items():
    spike_times, spike_neurons, t = simulate_snn_with_quantum_influence(scenario, tau)
    results[scenario] = {
        'spike_times': spike_times,
        'spike_neurons': spike_neurons,
        't': t,
        'tau': tau
    }

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Raster plots
for i, (scenario, data) in enumerate(results.items()):
    ax = axes[0, i] if i < 2 else axes[1, 0] if i == 2 else axes[1, 1]
    if len(data['spike_times']) > 0:
        ax.scatter(data['spike_times'], data['spike_neurons'], s=1, c='k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'{scenario}\n(τ = {data["tau"]:.1e} s)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, N_neurons)

# Plot 2: Firing rates
ax = axes[1, 1]
for scenario, data in results.items():
    if len(data['spike_times']) > 0:
        # Calculate firing rate over time
        bin_size = 0.1  # 100ms bins
        bins = np.arange(0, duration + bin_size, bin_size)
        rates, _ = np.histogram(data['spike_times'], bins=bins)
        rates = rates / (bin_size * N_neurons)  # Hz per neuron
        bin_centers = bins[:-1] + bin_size/2
        ax.plot(bin_centers, rates, label=f'{scenario} (τ={data["tau"]:.1e}s)')
    else:
        ax.plot([], [], label=f'{scenario} (τ={data["tau"]:.1e}s) - No spikes')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean firing rate (Hz)')
ax.set_title('Neural firing rates with quantum influence')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('snn_quantum_influence.png', dpi=200, bbox_inches='tight')

# --- Quantitative analysis ---
print("\nQuantitative analysis:")
for scenario, data in results.items():
    total_spikes = len(data['spike_times'])
    if total_spikes > 0:
        # Calculate firing rates in bins
        bin_size = 0.1  # 100ms bins
        bins = np.arange(0, duration + bin_size, bin_size)
        rates, _ = np.histogram(data['spike_times'], bins=bins)
        rates = rates / (bin_size * N_neurons)  # Hz per neuron
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        cv = std_rate / mean_rate if mean_rate > 0 else 0  # Coefficient of variation
        print(f"{scenario}: Total spikes = {total_spikes}, Mean rate = {mean_rate:.1f} ± {std_rate:.1f} Hz, CV = {cv:.2f}")
    else:
        print(f"{scenario}: Total spikes = {total_spikes}, Mean rate = 0.0 ± 0.0 Hz, CV = 0.00")

print("\nSaved figure: snn_quantum_influence.png")