#TODO check after nengo, not now

#import os

#for f in sorted(os.listdir(FIG_DIR)):
#    print(f)

""" This model measures gated memory robustness 

The purpose of this model is to investigate how chronic stress interacts with recurrent neural dynamics, excitatory/inhibitory (E/I) balance, and network topology, while he framework was designed as a controlled factorial simulation environment in which network architecture, recurrent cell type, and E/I dysregulation can be manipulated independently while preserving a biologically inspired chronic stress process.

Chronic stress is modeled not as transient noise alone, but as a slowly accumulating allostatic stress-load state. 
Different stress protocols should be investigated further, but maily to test the LSTM!

This internal stress state dynamically modulates recurrent gain, inhibitory efficacy, noise amplitude, and post-stress recovery behavior over time. The model therefore aims to capture persistent stress-induced state drift and reduced recovery capacity, which are characteristic of chronic stress pathology.

The framework compares multiple recurrent architectures, including vanilla recurrent neural networks (RNNs), gated recurrent units (GRUs), and long short-term memory networks (LSTMs), across dense, sparse, modular, and small-world connectivity regimes. In parallel, the model manipulates E/I balance through stress-dependent inhibitory loss and recurrent excitation gain changes.

The broader objective was to study how different recurrent memory mechanisms and circuit organizations influence resilience or vulnerability under prolonged stress exposure. Rather than functioning as a biophysically exact cortical simulation, the model serves as a computational psychiatry-inspired systems framework for exploring stress-related dynamical instability, recovery behavior, and persistent neural state drift.

# %% Step 1: Imports and global configuration

import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
# %% CELL 11: Optional matrix inspection

for name, res in results.items():
    model = res["model"]
    active, total, density = model.count_connections()
    W = model.recurrent_matrix_numpy()
    eigvals = np.linalg.eigvals(W)
    spectral_radius = np.max(np.abs(eigvals))

    print("\n", name)
    print("active connections:", active)
    print("total connections:", total)
    print("density:", round(density, 4))
    print("spectral radius:", round(float(spectral_radius), 4))
    print("positive edges:", int(np.sum(W > 0)))
    print("negative edges:", int(np.sum(W < 0)))


-------------- above not good enough-------------------------
Try make it more biological
This model is revised into a controlled factorial simulation framework. 
The chronic stress mechanism is preserved as a slow allostatic stress-load state that modulates noise amplitude, inhibitory efficacy, recurrent gain, and recovery dynamics. Which I currently identified as "important" for stress resilience. 
The main change is that network architecture, recurrent cell type, and E/I condition are now treated as independent experimental factors. 
This allows the model to compare dense, sparse, modular, and small-world recurrent architectures across vanilla RNN, GRU, and LSTM dynamics while separately manipulating E/I balance. 
The resulting design makes it possible to distinguish architecture effects from recurrent-memory effects and from excitatory/inhibitory dysregulation effects.
"""

#takes a while to run completely
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

#use seed, not real here
SEED = 072132
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FS = 10
DURATION_SEC = 180 #extend?
TIME_STEPS = FS * DURATION_SEC

#use 80 to 20 ratio for ei ratio
INPUT_DIM = 32
N_UNITS = 500
E_RATIO = 0.80
N_EXC = int(N_UNITS * E_RATIO)
N_INH = N_UNITS - N_EXC

BATCH_SIZE = 4
N_SEEDS = 5   # increase to 20 to 50 for stronger conclusions?

SAVE_DIR = "factorial_chronic_ei_results"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Time steps:", TIME_STEPS)
print("Excitatory units:", N_EXC)
print("Inhibitory units:", N_INH) 
# %% Step 2: Synthetic input and chronic stress protocol

#TODO try the smoother stress protocol, bursting stress protocol, too

def make_synthetic_input(batch_size, time_steps, input_dim, device="cpu"):
    x = torch.randn(batch_size, time_steps, input_dim, device=device) * 0.2

    t = torch.linspace(0, 2 * math.pi, time_steps, device=device)
    x[:, :, 0:1] += torch.sin(t).view(1, time_steps, 1)
    x[:, :, 1:2] += torch.cos(t * 0.5).view(1, time_steps, 1)

    return x

#I used a more smooth in previous version, so adjust if needed
def make_chronic_stress_protocol(
    time_steps,
    fs=10,
    pre_sec=45,
    ramp_sec=75,
    plateau_sec=30,
    stress_peak=1.0,
    tau_load_up_sec=20,
    tau_load_down_sec=90,
    device="cpu",
):
    pre_steps = int(pre_sec * fs)
    ramp_steps = int(ramp_sec * fs)
    plateau_steps = int(plateau_sec * fs)
    peri_steps = ramp_steps + plateau_steps
    post_steps = time_steps - pre_steps - peri_steps

    if post_steps <= 0:
        raise ValueError("Stress schedule exceeds simulation length.")

    external = torch.zeros(time_steps, device=device)

    external[pre_steps:pre_steps + ramp_steps] = torch.linspace(
        0.0, stress_peak, ramp_steps, device=device
    )
    external[pre_steps + ramp_steps:pre_steps + peri_steps] = stress_peak

    load = torch.zeros(time_steps, device=device)
    dt = 1.0 / fs

    for t in range(1, time_steps):
        target = external[t]
        tau = tau_load_up_sec if target > load[t - 1] else tau_load_down_sec
        alpha = dt / tau
        load[t] = load[t - 1] + alpha * (target - load[t - 1])

    phases = {
        "pre": (0, pre_steps),
        "peri": (pre_steps, pre_steps + peri_steps),
        "post": (pre_steps + peri_steps, time_steps),
    }

    return external, load, phases


x = make_synthetic_input(BATCH_SIZE, TIME_STEPS, INPUT_DIM, DEVICE)
external_stress, stress_load, phases = make_chronic_stress_protocol(
    TIME_STEPS, fs=FS, device=DEVICE
)

plt.figure(figsize=(12, 4))
plt.plot(external_stress.cpu(), label="external stress")
plt.plot(stress_load.cpu(), label="internal stress load")
plt.axvspan(*phases["pre"], alpha=0.08, label="pre")
plt.axvspan(*phases["peri"], alpha=0.08, label="peri")
plt.axvspan(*phases["post"], alpha=0.08, label="post")
plt.title("Chronic Stress Protocol")
plt.xlabel("time step")
plt.ylabel("stress level")
plt.legend()
plt.tight_layout()
plt.show()

# %% step 3: Architecture masks

def dense_mask(n_units):
    mask = torch.ones(n_units, n_units)
    mask.fill_diagonal_(0.0)
    return mask


def sparse_mask(n_units, density=0.10):
    mask = (torch.rand(n_units, n_units) < density).float()
    mask.fill_diagonal_(0.0)
    return mask


def modular_mask(n_units, n_modules=5, within_density=0.25, between_density=0.03):
    mask = torch.zeros(n_units, n_units)
    module_size = n_units // n_modules

    for i in range(n_modules):
        start_i = i * module_size
        end_i = n_units if i == n_modules - 1 else (i + 1) * module_size

        for j in range(n_modules):
            start_j = j * module_size
            end_j = n_units if j == n_modules - 1 else (j + 1) * module_size

            density = within_density if i == j else between_density
            block = (torch.rand(end_i - start_i, end_j - start_j) < density).float()
            mask[start_i:end_i, start_j:end_j] = block

    mask.fill_diagonal_(0.0)
    return mask


def small_world_mask(n_units, k=20, rewire_prob=0.10):
    mask = torch.zeros(n_units, n_units)

    half_k = k // 2
    for i in range(n_units):
        for offset in range(1, half_k + 1):
            j1 = (i + offset) % n_units
            j2 = (i - offset) % n_units
            mask[i, j1] = 1.0
            mask[i, j2] = 1.0

    for i in range(n_units):
        existing = torch.where(mask[i] > 0)[0].tolist()
        for j in existing:
            if random.random() < rewire_prob:
                mask[i, j] = 0.0
                new_j = random.randrange(n_units)
                while new_j == i:
                    new_j = random.randrange(n_units)
                mask[i, new_j] = 1.0

    mask.fill_diagonal_(0.0)
    return mask


def make_architecture_mask(architecture, n_units):
    if architecture == "dense":
        return dense_mask(n_units)
    if architecture == "sparse":
        return sparse_mask(n_units, density=0.10)
    if architecture == "modular":
        return modular_mask(n_units)
    if architecture == "small_world":
        return small_world_mask(n_units)
    raise ValueError(f"Unknown architecture: {architecture}")

# %% Step 4: Factorial ei recurrent model with RNN, GRU, and LSTM cells

class FactorialEIRecurrentNet(nn.Module):
    def __init__(
        self,
        input_dim=32,
        n_units=500,
        e_ratio=0.8,
        architecture="sparse",
        cell_type="rnn",
        ei_condition="baseline",
        noise_base=0.01,
        noise_stress=0.30,
        recurrent_gain_base=0.85,
        recurrent_gain_stress=0.10,
        leak_base=0.12,
        leak_stress=0.20,
        post_retention_base=0.00,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_units = n_units
        self.n_exc = int(n_units * e_ratio)
        self.n_inh = n_units - self.n_exc

        self.architecture = architecture
        self.cell_type = cell_type
        self.ei_condition = ei_condition

        self.noise_base = noise_base
        self.noise_stress = noise_stress
        self.recurrent_gain_base = recurrent_gain_base
        self.recurrent_gain_stress = recurrent_gain_stress
        self.leak_base = leak_base
        self.leak_stress = leak_stress
        self.post_retention_base = post_retention_base

        mask = make_architecture_mask(architecture, n_units)
        self.register_buffer("mask", mask)

        signs = torch.ones(n_units)
        signs[self.n_exc:] = -1.0
        self.register_buffer("pre_signs", signs.view(1, -1))

        self.input_proj = nn.Linear(input_dim, n_units)
        self.W_raw = nn.Parameter(torch.randn(n_units, n_units) * 0.03)
        self.bias = nn.Parameter(torch.zeros(n_units))

        if cell_type == "gru":
            self.gru_cell = nn.GRUCell(n_units, n_units)
        elif cell_type == "lstm":
            self.lstm_cell = nn.LSTMCell(n_units, n_units)
        elif cell_type != "rnn":
            raise ValueError("cell_type must be rnn, gru, or lstm")

        self.freezing_head = nn.Linear(n_units, 1)
        self.reward_head = nn.Linear(n_units, 1)

    def ei_parameters(self, load_t):
        if self.ei_condition == "baseline":
            inhibition_loss = 0.10
            gain_boost = 0.04
        elif self.ei_condition == "inhibition_loss":
            inhibition_loss = 0.70
            gain_boost = 0.04
        elif self.ei_condition == "gain_increase":
            inhibition_loss = 0.10
            gain_boost = 0.35
        elif self.ei_condition == "combined":
            inhibition_loss = 0.70
            gain_boost = 0.35
        else:
            raise ValueError(f"Unknown ei condition: {self.ei_condition}")

        inhibition_eff = 1.0 * (1.0 - inhibition_loss * load_t)
        inhibition_eff = torch.clamp(inhibition_eff, 0.05, 2.0)

        recurrent_gain = self.recurrent_gain_base + gain_boost * load_t

        return inhibition_eff, recurrent_gain

    def effective_weight(self, inhibition_eff):
        W_abs = torch.abs(self.W_raw) * self.mask
        signs = self.pre_signs.clone()
        signs[:, self.n_exc:] *= inhibition_eff
        return W_abs * signs

    def forward(
        self,
        x,
        stress_load=None,
        external_stress=None,
        fixed_noise=None,
    ):
        batch, time_steps, _ = x.shape
        device = x.device

        if stress_load is None:
            stress_load = torch.zeros(time_steps, device=device)
        if external_stress is None:
            external_stress = torch.zeros(time_steps, device=device)
        if fixed_noise is None:
            fixed_noise = torch.randn_like(x)

        h = torch.zeros(batch, self.n_units, device=device)
        c = torch.zeros(batch, self.n_units, device=device)

        states = []
        freezing = []
        reward = []
        ei_balance = []
        inhibition_trace = []
        gain_trace = []
        sigma_trace = []
        leak_trace = []

        for t in range(time_steps):
            load_t = stress_load[t].clamp(0.0, 1.0)
            external_t = external_stress[t].clamp(0.0, 1.0)

            sigma_t = self.noise_base + self.noise_stress * load_t
            noisy_input = x[:, t, :] + sigma_t * fixed_noise[:, t, :]

            inhibition_eff, recurrent_gain = self.ei_parameters(load_t)
            W_eff = self.effective_weight(inhibition_eff)

            u = self.input_proj(noisy_input)
            rec = torch.matmul(h, W_eff.T) * recurrent_gain
            drive = torch.tanh(u + rec + self.bias)

            leak_t = self.leak_base * (1.0 - self.leak_stress * load_t)
            leak_t = torch.clamp(leak_t, 0.02, 0.50)

            if self.cell_type == "rnn":
                h_candidate = drive
                h = (1.0 - leak_t) * h + leak_t * h_candidate

            elif self.cell_type == "gru":
                h_candidate = self.gru_cell(drive, h)
                h = (1.0 - leak_t) * h + leak_t * h_candidate

            elif self.cell_type == "lstm":
                h_candidate, c = self.lstm_cell(drive, (h, c))
                h = (1.0 - leak_t) * h + leak_t * h_candidate

            # Post-stress persistence only when external stress is off but load remains.
            post_persistence = (1.0 - external_t) * load_t * self.post_retention_base
            h = h * (1.0 + post_persistence)

            E_state = h[:, :self.n_exc]
            I_state = h[:, self.n_exc:]

            ei = E_state.abs().mean(dim=1, keepdim=True) - I_state.abs().mean(dim=1, keepdim=True)

            freeze_t = torch.sigmoid(self.freezing_head(h))
            reward_t = torch.sigmoid(self.reward_head(h) - freeze_t)

            states.append(h.unsqueeze(1))
            freezing.append(freeze_t.unsqueeze(1))
            reward.append(reward_t.unsqueeze(1))
            ei_balance.append(ei.unsqueeze(1))

            inhibition_trace.append(float(inhibition_eff.detach().cpu()))
            gain_trace.append(float(recurrent_gain.detach().cpu()))
            sigma_trace.append(float(sigma_t.detach().cpu()))
            leak_trace.append(float(leak_t.detach().cpu()))

        return {
            "states": torch.cat(states, dim=1),
            "freezing": torch.cat(freezing, dim=1),
            "reward": torch.cat(reward, dim=1),
            "ei_balance": torch.cat(ei_balance, dim=1),
            "inhibition_trace": np.array(inhibition_trace),
            "gain_trace": np.array(gain_trace),
            "sigma_trace": np.array(sigma_trace),
            "leak_trace": np.array(leak_trace),
        }

    def count_connections(self):
        active = int(self.mask.sum().item())
        total = int(self.mask.numel())
        density = active / total
        return active, total, density

    def recurrent_matrix_numpy(self, inhibition_eff=1.0):
        return self.effective_weight(torch.tensor(inhibition_eff, device=self.mask.device)).detach().cpu().numpy()

# %% step 5: Metrics

def compute_drift(clean_states, stress_states):
    return np.mean(np.abs(clean_states - stress_states), axis=(0, 2))


def compute_phase_metrics(clean, stress, phases):
    clean_states = clean["states"].detach().cpu().numpy()
    stress_states = stress["states"].detach().cpu().numpy()

    clean_freezing = clean["freezing"].detach().cpu().numpy()
    stress_freezing = stress["freezing"].detach().cpu().numpy()

    clean_reward = clean["reward"].detach().cpu().numpy()
    stress_reward = stress["reward"].detach().cpu().numpy()

    stress_ei = stress["ei_balance"].detach().cpu().numpy()

    drift_t = compute_drift(clean_states, stress_states)

    rows = []
    for phase in ["pre", "peri", "post"]:
        s, e = phases[phase]

        rows.append({
            "phase": phase,
            "state_drift_mean": float(drift_t[s:e].mean()),
            "state_drift_max": float(drift_t[s:e].max()),
            "freezing_clean_mean": float(clean_freezing[:, s:e, :].mean()),
            "freezing_stress_mean": float(stress_freezing[:, s:e, :].mean()),
            "reward_clean_mean": float(clean_reward[:, s:e, :].mean()),
            "reward_stress_mean": float(stress_reward[:, s:e, :].mean()),
            "ei_balance_stress_mean": float(stress_ei[:, s:e, :].mean()),
        })

    df = pd.DataFrame(rows)

    pre = df.loc[df.phase == "pre", "state_drift_mean"].values[0]
    peri = df.loc[df.phase == "peri", "state_drift_mean"].values[0]
    post = df.loc[df.phase == "post", "state_drift_mean"].values[0]

    summary = {
        "pre_drift": pre,
        "peri_drift": peri,
        "post_drift": post,
        "recovery_ratio_post_over_peri": float(post / (peri + 1e-8)),
        "persistent_drift_post_minus_pre": float(post - pre),
    }

    return df, summary, drift_t

# %% step 6: Factorial experiment definition

architectures = ["dense", "sparse", "modular", "small_world"]
cell_types = ["rnn", "gru", "lstm"]
ei_conditions = ["baseline", "inhibition_loss", "gain_increase", "combined"]

factorial_design = list(itertools.product(architectures, cell_types, ei_conditions))

print("Total conditions:", len(factorial_design))
print("Total runs:", len(factorial_design) * N_SEEDS)


# %% step 7: Run factorial simulations

all_summary_rows = []
example_results = {}

for seed in range(N_SEEDS):
    torch.manual_seed(SEED + seed)
    np.random.seed(SEED + seed)
    random.seed(SEED + seed)

    fixed_noise = torch.randn_like(x)

    for architecture, cell_type, ei_condition in factorial_design:
        name = f"{architecture}_{cell_type}_{ei_condition}_seed{seed}"
        print("Running:", name)

        model = FactorialEIRecurrentNet(
            input_dim=INPUT_DIM,
            n_units=N_UNITS,
            e_ratio=E_RATIO,
            architecture=architecture,
            cell_type=cell_type,
            ei_condition=ei_condition,
            noise_base=0.01,
            noise_stress=0.30,
            recurrent_gain_base=0.85,
            recurrent_gain_stress=0.10,
            leak_base=0.12,
            leak_stress=0.20,
            post_retention_base=0.02,
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():
            clean = model(
                x,
                stress_load=torch.zeros_like(stress_load),
                external_stress=torch.zeros_like(external_stress),
                fixed_noise=fixed_noise,
            )

            stress = model(
                x,
                stress_load=stress_load,
                external_stress=external_stress,
                fixed_noise=fixed_noise,
            )

        phase_df, summary, drift_t = compute_phase_metrics(clean, stress, phases)
        active, total, density = model.count_connections()

        row = {
            "run": name,
            "seed": seed,
            "architecture": architecture,
            "cell_type": cell_type,
            "ei_condition": ei_condition,
            "n_units": N_UNITS,
            "n_exc": N_EXC,
            "n_inh": N_INH,
            "active_connections": active,
            "total_connections": total,
            "density": density,
            **summary,
        }

        all_summary_rows.append(row)

        if seed == 0 and ei_condition == "combined":
            example_results[f"{architecture}_{cell_type}"] = {
                "clean": clean,
                "stress": stress,
                "drift_t": drift_t,
                "model": model,
            }

summary_df = pd.DataFrame(all_summary_rows)
summary_path = os.path.join(SAVE_DIR, "factorial_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("Saved:", summary_path)
summary_df.head()

# %% Step 7: Run factorial simulations

all_summary_rows = []
example_results = {}

for seed in range(N_SEEDS):
    torch.manual_seed(SEED + seed)
    np.random.seed(SEED + seed)
    random.seed(SEED + seed)

    fixed_noise = torch.randn_like(x)

    for architecture, cell_type, ei_condition in factorial_design:
        name = f"{architecture}_{cell_type}_{ei_condition}_seed{seed}"
        print("Running:", name)

        model = FactorialEIRecurrentNet(
            input_dim=INPUT_DIM,
            n_units=N_UNITS,
            e_ratio=E_RATIO,
            architecture=architecture,
            cell_type=cell_type,
            ei_condition=ei_condition,
            noise_base=0.01,
            noise_stress=0.30,
            recurrent_gain_base=0.85,
            recurrent_gain_stress=0.10,
            leak_base=0.12,
            leak_stress=0.20,
            post_retention_base=0.02,
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():
            clean = model(
                x,
                stress_load=torch.zeros_like(stress_load),
                external_stress=torch.zeros_like(external_stress),
                fixed_noise=fixed_noise,
            )

            stress = model(
                x,
                stress_load=stress_load,
                external_stress=external_stress,
                fixed_noise=fixed_noise,
            )

        phase_df, summary, drift_t = compute_phase_metrics(clean, stress, phases)
        active, total, density = model.count_connections()

        row = {
            "run": name,
            "seed": seed,
            "architecture": architecture,
            "cell_type": cell_type,
            "ei_condition": ei_condition,
            "n_units": N_UNITS,
            "n_exc": N_EXC,
            "n_inh": N_INH,
            "active_connections": active,
            "total_connections": total,
            "density": density,
            **summary,
        }

        all_summary_rows.append(row)

        if seed == 0 and ei_condition == "combined":
            example_results[f"{architecture}_{cell_type}"] = {
                "clean": clean,
                "stress": stress,
                "drift_t": drift_t,
                "model": model,
            }

summary_df = pd.DataFrame(all_summary_rows)
summary_path = os.path.join(SAVE_DIR, "factorial_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("Saved:", summary_path)
summary_df.head()

# %% Step 8: Aggregate factorial results

agg = (
    summary_df
    .groupby(["architecture", "cell_type", "ei_condition"])
    .agg(
        pre_drift_mean=("pre_drift", "mean"),
        peri_drift_mean=("peri_drift", "mean"),
        post_drift_mean=("post_drift", "mean"),
        recovery_ratio_mean=("recovery_ratio_post_over_peri", "mean"),
        persistent_drift_mean=("persistent_drift_post_minus_pre", "mean"),
        pre_drift_std=("pre_drift", "std"),
        peri_drift_std=("peri_drift", "std"),
        post_drift_std=("post_drift", "std"),
    )
    .reset_index()
)

agg_path = os.path.join(SAVE_DIR, "factorial_aggregate.csv")
agg.to_csv(agg_path, index=False)

agg.head(20)


# %% step 9: Architecture effect plot

plot_df = (
    summary_df
    .groupby("architecture")
    .agg(
        peri_drift=("peri_drift", "mean"),
        post_drift=("post_drift", "mean"),
        recovery_ratio=("recovery_ratio_post_over_peri", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(9, 4))
plt.bar(plot_df["architecture"], plot_df["post_drift"])
plt.ylabel("mean post-stress drift")
plt.title("Main Effect: Network Architecture")
plt.tight_layout()
plt.show()

plot_df

# %% step 10: Cell-type effect plot: RNN vs GRU vs LSTM

plot_df = (
    summary_df
    .groupby("cell_type")
    .agg(
        peri_drift=("peri_drift", "mean"),
        post_drift=("post_drift", "mean"),
        recovery_ratio=("recovery_ratio_post_over_peri", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(7, 4))
plt.bar(plot_df["cell_type"], plot_df["post_drift"])
plt.ylabel("mean post-stress drift")
plt.title("Main Effect: Recurrent Cell Type")
plt.tight_layout()
plt.show()

plot_df


# %% Step 11: ei condition effect plot

plot_df = (
    summary_df
    .groupby("ei_condition")
    .agg(
        peri_drift=("peri_drift", "mean"),
        post_drift=("post_drift", "mean"),
        recovery_ratio=("recovery_ratio_post_over_peri", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(9, 4))
plt.bar(plot_df["ei_condition"], plot_df["post_drift"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("mean post-stress drift")
plt.title("Main Effect: E/I Condition")
plt.tight_layout()
plt.show()

plot_df

# %% step 12: Interaction plot: architecture × cell type

interaction = (
    summary_df
    .groupby(["architecture", "cell_type"])
    .agg(post_drift=("post_drift", "mean"))
    .reset_index()
)

plt.figure(figsize=(10, 5))

for cell_type in cell_types:
    sub = interaction[interaction["cell_type"] == cell_type]
    plt.plot(
        sub["architecture"],
        sub["post_drift"],
        marker="o",
        linewidth=2.5,
        label=cell_type,
    )

plt.ylabel("mean post-stress drift")
plt.title("Interaction: Architecture × Cell Type")
plt.legend()
plt.tight_layout()
plt.show()

interaction


# %% Step 13: Interaction plot: cell type × ei condition

interaction = (
    summary_df
    .groupby(["cell_type", "ei_condition"])
    .agg(post_drift=("post_drift", "mean"))
    .reset_index()
)

plt.figure(figsize=(10, 5))

for cell_type in cell_types:
    sub = interaction[interaction["cell_type"] == cell_type]
    plt.plot(
        sub["ei_condition"],
        sub["post_drift"],
        marker="o",
        linewidth=2.5,
        label=cell_type,
    )

plt.xticks(rotation=20, ha="right")
plt.ylabel("mean post-stress drift")
plt.title("Interaction: Cell Type × E/I Condition")
plt.legend()
plt.tight_layout()
plt.show()

interaction

# %% step 14: Continuous drift examples for combined E/I condition

plt.figure(figsize=(13, 5))

for name, res in example_results.items():
    plt.plot(res["drift_t"], linewidth=2, label=name)

load_np = stress_load.detach().cpu().numpy()
load_scaled = load_np / (load_np.max() + 1e-8)

plt.plot(load_scaled, linestyle="--", linewidth=2.5, label="internal stress load")

plt.axvspan(*phases["pre"], alpha=0.08)
plt.axvspan(*phases["peri"], alpha=0.08)
plt.axvspan(*phases["post"], alpha=0.08)

plt.xlabel("time step")
plt.ylabel("brain-state drift")
plt.title("Continuous Drift: Combined E/I Dysregulation, Seed 0")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# %% step 15: Behavioral readouts for one selected example

selected = "sparse_lstm"

res = example_results[selected]
stress = res["stress"]

freezing = stress["freezing"].detach().cpu().numpy().squeeze()
reward = stress["reward"].detach().cpu().numpy().squeeze()
ei_balance = stress["ei_balance"].detach().cpu().numpy().squeeze()

plt.figure(figsize=(13, 4))
plt.plot(freezing.mean(axis=0), label="freezing-like output")
plt.plot(reward.mean(axis=0), label="reward-like output")
plt.plot(load_scaled, linestyle="--", label="stress load")
plt.axvspan(*phases["pre"], alpha=0.08)
plt.axvspan(*phases["peri"], alpha=0.08)
plt.axvspan(*phases["post"], alpha=0.08)
plt.title(f"Behavioral Readouts: {selected}")
plt.xlabel("time step")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(13, 4))
plt.plot(ei_balance.mean(axis=0), label="E/I balance")
plt.plot(load_scaled, linestyle="--", label="stress load")
plt.axvspan(*phases["pre"], alpha=0.08)
plt.axvspan(*phases["peri"], alpha=0.08)
plt.axvspan(*phases["post"], alpha=0.08)
plt.title(f"E/I Balance: {selected}")
plt.xlabel("time step")
plt.legend()
plt.tight_layout()
plt.show()

# %% step 16: Stress-modulated parameter traces

selected = "sparse_lstm"
res = example_results[selected]
stress = res["stress"]

plt.figure(figsize=(13, 4))
plt.plot(stress["sigma_trace"], label="noise sigma")
plt.plot(stress["inhibition_trace"], label="inhibition efficacy")
plt.plot(stress["gain_trace"], label="recurrent gain")
plt.plot(stress["leak_trace"], label="leak")
plt.axvspan(*phases["pre"], alpha=0.08)
plt.axvspan(*phases["peri"], alpha=0.08)
plt.axvspan(*phases["post"], alpha=0.08)
plt.title(f"Stress-Modulated Parameters: {selected}")
plt.xlabel("time step")
plt.legend()
plt.tight_layout()
plt.show()

# %% step 17: Simple statistical summary

try:
    import statsmodels.formula.api as smf

    model = smf.ols(
        "post_drift ~ C(architecture) * C(cell_type) * C(ei_condition)",
        data=summary_df
    ).fit()

    print(model.summary())

except ImportError:
    print("statsmodels is not installed. Run:")
    print("!pip install statsmodels")

# %% step 18: Save final tables

summary_df.to_csv(os.path.join(SAVE_DIR, "factorial_summary.csv"), index=False)
agg.to_csv(os.path.join(SAVE_DIR, "factorial_aggregate.csv"), index=False)

print("Saved files:")
print(os.path.join(SAVE_DIR, "factorial_summary.csv"))
print(os.path.join(SAVE_DIR, "factorial_aggregate.csv"))

# %% step 19: Publication-style figure panel to see what is going on

#adjust path depending on coding env

FIG_DIR = os.path.join(SAVE_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

summary_df = pd.read_csv(os.path.join(SAVE_DIR, "factorial_summary.csv"))

# ---------- Helper functions ----------

#not well done, my nw pipeline is maybe better - but anyway I think this will not be used... maybe the implementation still can inform other projects
