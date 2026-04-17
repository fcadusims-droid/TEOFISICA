import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import warnings
warnings.filterwarnings("ignore")

# --- Parameters from previous quantum simulation ---
# Coherence times from the Bloch-Redfield simulation (in seconds)
coherence_times = {
    "lowfreq_bath": 3.55e-07,  # Low-frequency bath (Markov)
    "silent_gap": 3.55e-07,    # Protein phonon bath (BR)
    "floquet_zeno": 1.06e-07,  # High-frequency Floquet/Zeno (BR)
}

# Convert to dimensionless units (same as quantum sim)
time_unit = 3e-8  # s per unit
coherence_units = {k: v / time_unit for k, v in coherence_times.items()}

# --- Neural Network Parameters ---
N_neurons = 100  # Number of neurons in the network
duration = 1.0 * second  # Simulation duration
dt = 0.1 * ms  # Time step

# Leaky Integrate-and-Fire neuron parameters
tau_m = 20 * ms  # Membrane time constant
tau_ref = 2 * ms  # Refractory period
V_th = -50 * mV  # Threshold
V_reset = -65 * mV  # Reset potential
V_rest = -65 * mV  # Resting potential
I_bias = 1 * mV  # Bias current

# Synaptic parameters
tau_syn = 5 * ms  # Synaptic time constant
w_exc = 2 * mV  # Excitatory weight
w_inh = -2 * mV  # Inhibitory weight
p_conn = 0.1  # Connection probability

# External input
rate_ext = 50 * Hz  # External Poisson input rate
w_ext = 2 * mV  # External input weight

# --- Quantum influence function ---
def quantum_modulation(coherence_tau, t):
    """
    Modulate neural firing based on quantum coherence time.
    Longer coherence -> more stable firing patterns (less stochasticity).
    """
    # Effective modulation: coherence acts as a damping factor
    modulation = np.exp(-t / (coherence_tau * time_unit))
    return modulation

# --- Simulation function ---
def simulate_snn_with_quantum_influence(scenario_name, coherence_tau):
    print(f"Simulating SNN with {scenario_name} (tau = {coherence_tau:.2e} s)")

    # Neuron equations (Leaky Integrate-and-Fire model)
    eqs = '''
    dv/dt = (V_rest - v + I + I_bias + quantum_influence) / tau_m : volt
    dI/dt = -I / tau_syn : volt
    quantum_influence : volt
    '''

    # Reset and threshold
    reset = 'v = V_reset'

    threshold = 'v > V_th'

    # Create neuron group
    neurons = NeuronGroup(N_neurons, eqs, threshold=threshold, reset=reset, refractory=tau_ref, method='euler', dt=dt)

    # Initialize
    neurons.v = V_rest + (V_th - V_rest) * np.random.rand(N_neurons)

    # Synapses
    synapses_exc = Synapses(neurons, neurons, 'w : volt', on_pre='I_post += w', delay=1*ms)
    synapses_exc.connect(p=p_conn)
    synapses_exc.w = w_exc

    synapses_inh = Synapses(neurons, neurons, 'w : volt', on_pre='I_post += w', delay=1*ms)
    synapses_inh.connect(p=p_conn)
    synapses_inh.w = w_inh

    # External input
    input_group = PoissonGroup(N_neurons, rate_ext)
    input_synapses = Synapses(input_group, neurons, on_pre='I_post += w_ext')
    input_synapses.connect(j='i')  # Connect each input to corresponding neuron

    # Quantum influence: modulate based on coherence time
    @network_operation(dt=dt)
    def apply_quantum_influence():
        t_current = defaultclock.t / second  # in seconds
        modulation = quantum_modulation(coherence_tau, t_current)
        noise_level = 2 * mV * (1 - modulation)  # More noise when coherence is low
        neurons.quantum_influence = noise_level * np.random.randn(N_neurons)

    # Monitors
    spike_monitor = SpikeMonitor(neurons)
    rate_monitor = PopulationRateMonitor(neurons)

    # Network
    net = Network(neurons, synapses_exc, synapses_inh, input_group, input_synapses, spike_monitor, rate_monitor, apply_quantum_influence)
    net.run(duration, report='text')

    return spike_monitor, rate_monitor

# --- Run simulations for each quantum scenario ---
results = {}
for scenario, tau in coherence_times.items():
    spike_mon, rate_mon = simulate_snn_with_quantum_influence(scenario, tau)
    results[scenario] = {
        'spikes': spike_mon,
        'rates': rate_mon,
        'tau': tau
    }

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Raster plots
for i, (scenario, data) in enumerate(results.items()):
    ax = axes[0, i] if i < 2 else axes[1, 0] if i == 2 else axes[1, 1]
    ax.plot(data['spikes'].t / ms, data['spikes'].i, '.k', markersize=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'{scenario}\n(τ = {data["tau"]:.1e} s)')
    ax.set_xlim(0, duration / ms)
    ax.set_ylim(0, N_neurons)

# Plot 2: Firing rates
ax = axes[1, 1]
for scenario, data in results.items():
    ax.plot(data['rates'].t / ms, data['rates'].rate / Hz, label=f'{scenario} (τ={data["tau"]:.1e}s)')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Population firing rate (Hz)')
ax.set_title('Neural firing rates with quantum influence')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('snn_quantum_influence.png', dpi=200, bbox_inches='tight')

# --- Quantitative analysis ---
print("\nQuantitative analysis:")
for scenario, data in results.items():
    total_spikes = len(data['spikes'].t)
    mean_rate = np.mean(data['rates'].rate / Hz)
    std_rate = np.std(data['rates'].rate / Hz)
    print(f"{scenario}: Total spikes = {total_spikes}, Mean rate = {mean_rate:.1f} ± {std_rate:.1f} Hz")

print("\nSaved figure: snn_quantum_influence.png")
