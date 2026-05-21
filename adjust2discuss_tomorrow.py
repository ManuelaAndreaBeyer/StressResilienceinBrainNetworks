#%% Step 1: Imports and run configuration

import os
import math
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nengo

#TODO check this implementation on Friday and what is diffferent to the MNIST version whre I used a digit, waited either 50 or 100 ms and then the network had to learn
#TODO check this vs multi-tensor version, pattern learning in signal 
#reminder moving dot on image -> worth to search for or generate a test data set?
now = datetime.now()
RUN_STAMP = f"{now.strftime('%Y')}{now.strftime('%b').upper()}{now.strftime('%d_%H_%M')}"
RESULTS_DIR = Path("results") / RUN_STAMP
FIGURES_DIR = RESULTS_DIR / "figures"
STATISTICS_DIR = RESULTS_DIR / "statistics"
DATA_DIR = RESULTS_DIR / "data"

for directory in [RESULTS_DIR, FIGURES_DIR, STATISTICS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SEED = 72132
np.random.seed(SEED)
random.seed(SEED)

FS = 10
DT = 1.0 / FS
DURATION_SEC = 180
TIME_STEPS = FS * DURATION_SEC
T = np.arange(TIME_STEPS) * DT

N_UNITS = 500
E_RATIO = 0.80
N_EXC = int(N_UNITS * E_RATIO)
N_INH = N_UNITS - N_EXC

INPUT_DIM = 32
N_SEEDS = 5 #was higher in later versions

ARCHITECTURES = ["dense", "sparse", "modular", "small_world"]
PHENOTYPES = ["control", "resilient", "vulnerable_depressive"] 
#idea was to have later other than depressive, but for PhD depressive (I invested too much time + depression and PTSD-like, traume fits Ca-Imaging)

#TODO discuss the perturbations with the nengo team 
#i used white noise in a different context it does NOT fit here
PERTURBATIONS = [
    "none",
    "gaussian_noise",
    "residual_degradation",
    "synaptic_gain_amplification",
    "heterogeneous_time_constants",
    "partial_neuronal_silencing",
    "recurrent_instability_induction",
]

PRIMARY_PERTURBATIONS = [
    "none",
    "gaussian_noise",
    "residual_degradation",
    "synaptic_gain_amplification",
    "heterogeneous_time_constants",
    "partial_neuronal_silencing",
    "recurrent_instability_induction",
]

print("Run stamp:", RUN_STAMP)
print("Results directory:", RESULTS_DIR)


#%% Step 2: Save helpers

def save_figure(name):
    png_path = FIGURES_DIR / f"{name}_{RUN_STAMP}.png"
    svg_path = FIGURES_DIR / f"{name}_{RUN_STAMP}.svg"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", svg_path)


def save_dataframe(df, name):
    path = STATISTICS_DIR / f"{name}_{RUN_STAMP}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)


def save_numpy(array, name):
    path = DATA_DIR / f"{name}_{RUN_STAMP}.npy"
    np.save(path, array)
    print("Saved:", path)


#%% Step 3: Synthetic input and chronic stress protocol

def make_synthetic_input(time_steps, input_dim, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.15, size=(time_steps, input_dim)).astype(np.float32)
    t = np.linspace(0, 2 * np.pi, time_steps)
    x[:, 0] += np.sin(t)
    x[:, 1] += np.cos(0.5 * t)
    x[:, 2] += np.sin(2.0 * t) * 0.4
    x[:, 3] += np.linspace(-0.5, 0.5, time_steps)
    return x


def make_chronic_stress_protocol(
    time_steps,
    fs=10,
    pre_sec=45,
    ramp_sec=75,
    plateau_sec=30,
    stress_peak=1.0,
    tau_load_up_sec=20,
    tau_load_down_sec=90,
):
    pre_steps = int(pre_sec * fs)
    ramp_steps = int(ramp_sec * fs)
    plateau_steps = int(plateau_sec * fs)
    peri_steps = ramp_steps + plateau_steps
    post_steps = time_steps - pre_steps - peri_steps

    if post_steps <= 0:
        raise ValueError("Stress schedule exceeds simulation length.")

    external = np.zeros(time_steps, dtype=np.float32)
    external[pre_steps:pre_steps + ramp_steps] = np.linspace(0.0, stress_peak, ramp_steps)
    external[pre_steps + ramp_steps:pre_steps + peri_steps] = stress_peak

    load = np.zeros(time_steps, dtype=np.float32)
    dt = 1.0 / fs

    for i in range(1, time_steps):
        target = external[i]
        tau = tau_load_up_sec if target > load[i - 1] else tau_load_down_sec
        alpha = dt / tau
        load[i] = load[i - 1] + alpha * (target - load[i - 1])

    phases = {
        "pre": (0, pre_steps),
        "peri": (pre_steps, pre_steps + peri_steps),
        "post": (pre_steps + peri_steps, time_steps),
    }

    return external, load, phases


external_stress, stress_load, phases = make_chronic_stress_protocol(TIME_STEPS, FS)

plt.figure(figsize=(12, 4))
plt.plot(T, external_stress, label="external stress")
plt.plot(T, stress_load, label="internal stress load")
for phase, (s, e) in phases.items():
    plt.axvspan(T[s], T[e - 1], alpha=0.08, label=phase)
plt.title("Chronic stress protocol")
plt.xlabel("time (s)")
plt.ylabel("stress level")
plt.legend()
save_figure("step3_chronic_stress_protocol")
plt.show()

save_numpy(external_stress, "external_stress")
save_numpy(stress_load, "stress_load")


#%% Step 4: Connectivity masks

def dense_mask(n_units):
    mask = np.ones((n_units, n_units), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def sparse_mask(n_units, density=0.10, seed=42):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n_units, n_units)) < density).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def modular_mask(n_units, n_modules=5, within_density=0.25, between_density=0.03, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.zeros((n_units, n_units), dtype=np.float32)
    module_size = n_units // n_modules

    for i in range(n_modules):
        start_i = i * module_size
        end_i = n_units if i == n_modules - 1 else (i + 1) * module_size

        for j in range(n_modules):
            start_j = j * module_size
            end_j = n_units if j == n_modules - 1 else (j + 1) * module_size

            density = within_density if i == j else between_density
            block = (rng.random((end_i - start_i, end_j - start_j)) < density).astype(np.float32)
            mask[start_i:end_i, start_j:end_j] = block

    np.fill_diagonal(mask, 0.0)
    return mask


def small_world_mask(n_units, k=20, rewire_prob=0.10, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.zeros((n_units, n_units), dtype=np.float32)
    half_k = k // 2

    for i in range(n_units):
        for offset in range(1, half_k + 1):
            mask[i, (i + offset) % n_units] = 1.0
            mask[i, (i - offset) % n_units] = 1.0

    for i in range(n_units):
        existing = np.where(mask[i] > 0)[0].tolist()
        for j in existing:
            if rng.random() < rewire_prob:
                mask[i, j] = 0.0
                new_j = int(rng.integers(0, n_units))
                while new_j == i:
                    new_j = int(rng.integers(0, n_units))
                mask[i, new_j] = 1.0

    np.fill_diagonal(mask, 0.0)
    return mask


def make_architecture_mask(architecture, n_units, seed):
    if architecture == "dense":
        return dense_mask(n_units)
    if architecture == "sparse":
        return sparse_mask(n_units, density=0.10, seed=seed)
    if architecture == "modular":
        return modular_mask(n_units, seed=seed)
    if architecture == "small_world":
        return small_world_mask(n_units, seed=seed)
    raise ValueError(architecture)


#%% Step 5: E/I residual ResNet dynamics

class EIResidualResNetDynamics:
    def __init__(
        self,
        architecture,
        phenotype,
        perturbation,
        seed,
        n_units=N_UNITS,
        input_dim=INPUT_DIM,
        e_ratio=E_RATIO,
    ):
        self.architecture = architecture
        self.phenotype = phenotype
        self.perturbation = perturbation
        self.seed = seed
        self.n_units = n_units
        self.input_dim = input_dim
        self.e_ratio = e_ratio
        self.n_exc = int(n_units * e_ratio)
        self.n_inh = n_units - self.n_exc

        self.rng = np.random.default_rng(seed)
        self.mask = make_architecture_mask(architecture, n_units, seed)

        self.input_projection = self.rng.normal(0.0, 0.12, size=(n_units, input_dim)).astype(np.float32)
        self.W_raw = self.rng.normal(0.0, 0.035, size=(n_units, n_units)).astype(np.float32)

        self.residual_scale = 0.80
        self.base_leak = 0.12
        self.base_noise = 0.01
        self.base_gain = 0.82
        self.inhibition_loss_strength = 0.10
        self.gain_increase_strength = 0.04
        self.noise_stress = 0.08
        self.leak_stress = 0.12
        self.post_retention = 0.00

        if phenotype == "control":
            self.inhibition_loss_strength = 0.04
            self.gain_increase_strength = 0.02
            self.noise_stress = 0.03
            self.leak_stress = 0.05
            self.post_retention = 0.00
            self.residual_scale = 0.90

        if phenotype == "resilient":
            self.inhibition_loss_strength = 0.12
            self.gain_increase_strength = 0.06
            self.noise_stress = 0.10
            self.leak_stress = 0.08
            self.post_retention = 0.01
            self.residual_scale = 0.95

        if phenotype == "vulnerable_depressive":
            self.inhibition_loss_strength = 0.70
            self.gain_increase_strength = 0.35
            self.noise_stress = 0.30
            self.leak_stress = 0.20
            self.post_retention = 0.04
            self.residual_scale = 0.65

        self.silence_mask = np.ones(n_units, dtype=np.float32)
        self.tau_vector = np.ones(n_units, dtype=np.float32)

        if perturbation == "residual_degradation":
            self.residual_scale *= 0.35

        if perturbation == "synaptic_gain_amplification":
            self.gain_increase_strength += 0.25

        if perturbation == "heterogeneous_time_constants":
            self.tau_vector = self.rng.uniform(0.55, 1.80, size=n_units).astype(np.float32)

        if perturbation == "partial_neuronal_silencing":
            silenced = self.rng.choice(n_units, size=int(0.15 * n_units), replace=False)
            self.silence_mask[silenced] = 0.0

        if perturbation == "recurrent_instability_induction":
            self.base_gain += 0.20
            self.gain_increase_strength += 0.20
            self.inhibition_loss_strength += 0.15

        signs = np.ones(n_units, dtype=np.float32)
        signs[self.n_exc:] = -1.0
        self.pre_signs = signs.reshape(1, -1)

        self.last_index = -1
        self.h = np.zeros(n_units, dtype=np.float32)
        self.records = {
            "state": [],
            "ei_balance": [],
            "exc_activity": [],
            "inh_activity": [],
            "gain": [],
            "inhibition_eff": [],
            "sigma": [],
            "leak": [],
            "residual_scale": [],
            "freezing_like": [],
            "reward_like": [],
        }

    def reset(self):
        self.last_index = -1
        self.h = np.zeros(self.n_units, dtype=np.float32)
        for key in self.records:
            self.records[key] = []

    def effective_weight(self, inhibition_eff):
        W_abs = np.abs(self.W_raw) * self.mask
        signs = self.pre_signs.copy()
        signs[:, self.n_exc:] *= inhibition_eff
        return (W_abs * signs).astype(np.float32)

    def step(self, t, x_t, external_t, load_t, stressed=True):
        index = min(int(round(t * FS)), TIME_STEPS - 1)

        if index <= self.last_index:
            index = self.last_index + 1

        if index >= TIME_STEPS:
            return 0.0

        self.last_index = index

        load = float(load_t) if stressed else 0.0
        external = float(external_t) if stressed else 0.0

        inhibition_eff = float(np.clip(1.0 - self.inhibition_loss_strength * load, 0.05, 2.0))
        recurrent_gain = float(self.base_gain + self.gain_increase_strength * load)
        sigma = float(self.base_noise + self.noise_stress * load)

        if self.perturbation == "gaussian_noise":
            sigma += 0.20 * max(load, 0.05)

        leak = float(np.clip(self.base_leak * (1.0 - self.leak_stress * load), 0.02, 0.50))
        leak_vector = np.clip(leak * self.tau_vector, 0.01, 0.70).astype(np.float32)

        noise = self.rng.normal(0.0, sigma, size=self.input_dim).astype(np.float32)
        input_drive = self.input_projection @ (x_t + noise)

        W_eff = self.effective_weight(inhibition_eff)
        recurrent_drive = W_eff @ self.h
        transformed = np.tanh(input_drive + recurrent_gain * recurrent_drive)

        residual = self.residual_scale * self.h

        if self.perturbation == "residual_degradation":
            residual_noise = self.rng.normal(0.0, 0.08 + 0.12 * load, size=self.n_units).astype(np.float32)
            residual = residual + residual_noise

        next_h = residual + (1.0 - self.residual_scale) * transformed
        next_h = (1.0 - leak_vector) * self.h + leak_vector * next_h

        post_persistence = (1.0 - external) * load * self.post_retention
        next_h = next_h * (1.0 + post_persistence)
        next_h = next_h * self.silence_mask
        next_h = np.tanh(next_h).astype(np.float32)

        self.h = next_h

        exc = np.mean(np.abs(self.h[:self.n_exc]))
        inh = np.mean(np.abs(self.h[self.n_exc:]))
        ei_balance = exc - inh

        freezing_like = 1.0 / (1.0 + np.exp(-4.0 * (ei_balance + np.mean(self.h) - 0.05)))
        reward_like = 1.0 / (1.0 + np.exp(4.0 * (ei_balance + np.mean(self.h) - 0.05)))

        self.records["state"].append(self.h.copy())
        self.records["ei_balance"].append(ei_balance)
        self.records["exc_activity"].append(exc)
        self.records["inh_activity"].append(inh)
        self.records["gain"].append(recurrent_gain)
        self.records["inhibition_eff"].append(inhibition_eff)
        self.records["sigma"].append(sigma)
        self.records["leak"].append(leak)
        self.records["residual_scale"].append(self.residual_scale)
        self.records["freezing_like"].append(float(freezing_like))
        self.records["reward_like"].append(float(reward_like))

        return float(np.mean(self.h))

    def as_arrays(self):
        out = {}
        for key, value in self.records.items():
            out[key] = np.asarray(value)
        return out


#%% Step 6: Nengo model builder

def build_nengo_resnet_model(dynamics, x_input, external_stress, stress_load, stressed=True):
    model = nengo.Network(label=f"{dynamics.architecture}_{dynamics.phenotype}_{dynamics.perturbation}")

    with model:
        input_node = nengo.Node(
            lambda t: x_input[min(int(round(t * FS)), TIME_STEPS - 1)],
            size_out=INPUT_DIM,
            label="input_signal",
        )

        external_node = nengo.Node(
            lambda t: external_stress[min(int(round(t * FS)), TIME_STEPS - 1)],
            size_out=1,
            label="external_stress",
        )

        load_node = nengo.Node(
            lambda t: stress_load[min(int(round(t * FS)), TIME_STEPS - 1)],
            size_out=1,
            label="internal_stress_load",
        )

        combined_node = nengo.Node(
            lambda t, x: dynamics.step(
                t=t,
                x_t=x[:INPUT_DIM],
                external_t=x[INPUT_DIM],
                load_t=x[INPUT_DIM + 1],
                stressed=stressed,
            ),
            size_in=INPUT_DIM + 2,
            size_out=1,
            label="ei_residual_resnet_dynamics",
        )

        output_probe = nengo.Probe(combined_node, synapse=None)

        nengo.Connection(input_node, combined_node[:INPUT_DIM], synapse=None)
        nengo.Connection(external_node, combined_node[INPUT_DIM], synapse=None)
        nengo.Connection(load_node, combined_node[INPUT_DIM + 1], synapse=None)

    return model, output_probe


#%% Step 7: Metric functions


#! Metric here not yet the actual NW metrics

#TODO include the network metrics analysis after Nengo !!! just for some tried 


def align_length(a, target_len):
    if len(a) == target_len:
        return a
    if len(a) > target_len:
        return a[:target_len]
    if len(a) == 0:
        return np.zeros(target_len)
    pad = np.repeat(a[-1:], target_len - len(a), axis=0)
    return np.concatenate([a, pad], axis=0)


def compute_phase_metrics(clean_records, stress_records, phases):
    clean_states = align_length(clean_records["state"], TIME_STEPS)
    stress_states = align_length(stress_records["state"], TIME_STEPS)

    drift_t = np.mean(np.abs(stress_states - clean_states), axis=1)

    rows = []
    for phase, (s, e) in phases.items():
        rows.append({
            "phase": phase,
            "state_drift_mean": float(np.mean(drift_t[s:e])),
            "state_drift_max": float(np.max(drift_t[s:e])),
            "ei_balance_mean": float(np.mean(align_length(stress_records["ei_balance"], TIME_STEPS)[s:e])),
            "exc_activity_mean": float(np.mean(align_length(stress_records["exc_activity"], TIME_STEPS)[s:e])),
            "inh_activity_mean": float(np.mean(align_length(stress_records["inh_activity"], TIME_STEPS)[s:e])),
            "gain_mean": float(np.mean(align_length(stress_records["gain"], TIME_STEPS)[s:e])),
            "inhibition_eff_mean": float(np.mean(align_length(stress_records["inhibition_eff"], TIME_STEPS)[s:e])),
            "sigma_mean": float(np.mean(align_length(stress_records["sigma"], TIME_STEPS)[s:e])),
            "leak_mean": float(np.mean(align_length(stress_records["leak"], TIME_STEPS)[s:e])),
            "freezing_like_mean": float(np.mean(align_length(stress_records["freezing_like"], TIME_STEPS)[s:e])),
            "reward_like_mean": float(np.mean(align_length(stress_records["reward_like"], TIME_STEPS)[s:e])),
        })

    phase_df = pd.DataFrame(rows)

    pre = phase_df.loc[phase_df.phase == "pre", "state_drift_mean"].iloc[0]
    peri = phase_df.loc[phase_df.phase == "peri", "state_drift_mean"].iloc[0]
    post = phase_df.loc[phase_df.phase == "post", "state_drift_mean"].iloc[0]

    summary = {
        "pre_drift": float(pre),
        "peri_drift": float(peri),
        "post_drift": float(post),
        "recovery_ratio_post_over_peri": float(post / (peri + 1e-8)),
        "persistent_drift_post_minus_pre": float(post - pre),
        "phase_transition_index": float(np.max(np.gradient(drift_t))),
        "drift_auc": float(np.trapz(drift_t, dx=DT)),
    }

    return phase_df, summary, drift_t


def spectral_radius_estimate(dynamics):
    W = dynamics.effective_weight(1.0)
    sample = W[::5, ::5]
    eigvals = np.linalg.eigvals(sample)
    return float(np.max(np.abs(eigvals)))


#%% Step 8: Plotting functions 
# 
#TODO discuss with Janina how to imporive

def plot_timeseries(time, values, title, ylabel, name):
    plt.figure(figsize=(12, 4))
    plt.plot(time, values)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    save_figure(name)
    plt.show()


def plot_group_bar(df, x_col, y_col, group_col, title, name):
    pivot = df.groupby([x_col, group_col])[y_col].mean().reset_index()
    labels = list(pivot[x_col].unique())
    groups = list(pivot[group_col].unique())
    x = np.arange(len(labels))
    width = 0.8 / max(len(groups), 1)

    plt.figure(figsize=(12, 5))
    for i, group in enumerate(groups):
        vals = []
        for label in labels:
            subset = pivot[(pivot[x_col] == label) & (pivot[group_col] == group)]
            vals.append(float(subset[y_col].iloc[0]) if len(subset) else 0.0)
        plt.bar(x + i * width - width * (len(groups) - 1) / 2, vals, width, label=group)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(y_col)
    plt.legend()
    save_figure(name)
    plt.show()


#%% Step 9: Run one simulation pair

def run_pair(architecture, phenotype, perturbation, seed):
    x_input = make_synthetic_input(TIME_STEPS, INPUT_DIM, seed)

    clean_dynamics = EIResidualResNetDynamics(
        architecture=architecture,
        phenotype=phenotype,
        perturbation=perturbation,
        seed=seed,
    )

    stress_dynamics = EIResidualResNetDynamics(
        architecture=architecture,
        phenotype=phenotype,
        perturbation=perturbation,
        seed=seed,
    )

    clean_model, clean_probe = build_nengo_resnet_model(
        dynamics=clean_dynamics,
        x_input=x_input,
        external_stress=np.zeros_like(external_stress),
        stress_load=np.zeros_like(stress_load),
        stressed=False,
    )

    stress_model, stress_probe = build_nengo_resnet_model(
        dynamics=stress_dynamics,
        x_input=x_input,
        external_stress=external_stress,
        stress_load=stress_load,
        stressed=True,
    )

    with nengo.Simulator(clean_model, dt=DT, progress_bar=False) as sim:
        sim.run(DURATION_SEC)

    with nengo.Simulator(stress_model, dt=DT, progress_bar=False) as sim:
        sim.run(DURATION_SEC)

    clean_records = clean_dynamics.as_arrays()
    stress_records = stress_dynamics.as_arrays()

    phase_df, summary, drift_t = compute_phase_metrics(clean_records, stress_records, phases)

    active_connections = int(np.sum(stress_dynamics.mask))
    total_connections = int(stress_dynamics.mask.size)
    density = active_connections / total_connections
    spectral_radius = spectral_radius_estimate(stress_dynamics)

    summary_row = {
        "architecture": architecture,
        "phenotype": phenotype,
        "perturbation": perturbation,
        "seed": seed,
        "n_units": N_UNITS,
        "n_exc": N_EXC,
        "n_inh": N_INH,
        "e_ratio": E_RATIO,
        "active_connections": active_connections,
        "total_connections": total_connections,
        "density": density,
        "spectral_radius_estimate": spectral_radius,
        **summary,
    }

    phase_df["architecture"] = architecture
    phase_df["phenotype"] = phenotype
    phase_df["perturbation"] = perturbation
    phase_df["seed"] = seed

    return summary_row, phase_df, clean_records, stress_records, drift_t


#%% Step 10: Full factorial experiment

all_summary_rows = []
all_phase_rows = []
example_store = {}

for seed_offset in range(N_SEEDS):
    seed = SEED + seed_offset

    for architecture in ARCHITECTURES:
        for phenotype in PHENOTYPES:
            for perturbation in PRIMARY_PERTURBATIONS:
                print("Running:", architecture, phenotype, perturbation, "seed", seed)

                summary_row, phase_df, clean_records, stress_records, drift_t = run_pair(
                    architecture=architecture,
                    phenotype=phenotype,
                    perturbation=perturbation,
                    seed=seed,
                )

                all_summary_rows.append(summary_row)
                all_phase_rows.append(phase_df)

                key = f"{architecture}_{phenotype}_{perturbation}"
                if seed_offset == 0 and key not in example_store:
                    example_store[key] = {
                        "clean": clean_records,
                        "stress": stress_records,
                        "drift_t": drift_t,
                    }

summary_df = pd.DataFrame(all_summary_rows)
phase_df = pd.concat(all_phase_rows, ignore_index=True)

save_dataframe(summary_df, "factorial_summary_statistics") #keep?
save_dataframe(phase_df, "phase_resolved_statistics")


#%% Step 11: Main summary plots

plot_group_bar(
    summary_df,
    x_col="architecture",
    y_col="post_drift",
    group_col="phenotype",
    title="Post-stress drift by architecture and phenotype",
    name="step11_post_drift_by_architecture_phenotype",
)

plot_group_bar(
    summary_df,
    x_col="architecture",
    y_col="recovery_ratio_post_over_peri",
    group_col="phenotype",
    title="Recovery ratio by architecture and phenotype",
    name="step11_recovery_ratio_by_architecture_phenotype",
)

plot_group_bar(
    summary_df,
    x_col="architecture",
    y_col="phase_transition_index",
    group_col="phenotype",
    title="Phase transition index by architecture and phenotype",
    name="step11_phase_transition_index_by_architecture_phenotype",
)

plot_group_bar(
    summary_df,
    x_col="phenotype",
    y_col="post_drift",
    group_col="perturbation",
    title="Post-stress drift by phenotype and perturbation",
    name="step11_post_drift_by_phenotype_perturbation",
)


#%% Step 12: Example trajectory plots

for key, item in example_store.items():
    if key not in [
        "small_world_resilient_none",
        "small_world_vulnerable_depressive_none",
        "sparse_resilient_none",
        "dense_vulnerable_depressive_none",
        "small_world_vulnerable_depressive_residual_degradation",
        "small_world_vulnerable_depressive_recurrent_instability_induction",
    ]:
        continue

    clean = item["clean"]
    stress = item["stress"]
    drift_t = align_length(item["drift_t"], TIME_STEPS)

    plot_timeseries(
        T,
        drift_t,
        f"State drift: {key}",
        "mean absolute drift",
        f"step12_state_drift_{key}",
    )

    plot_timeseries(
        T,
        align_length(stress["ei_balance"], TIME_STEPS),
        f"E/I balance: {key}",
        "E activity - I activity",
        f"step12_ei_balance_{key}",
    )

    plot_timeseries(
        T,
        align_length(stress["freezing_like"], TIME_STEPS),
        f"Freezing-like output: {key}",
        "freezing-like output",
        f"step12_freezing_like_{key}",
    )

    plot_timeseries(
        T,
        align_length(stress["reward_like"], TIME_STEPS),
        f"Reward-like output: {key}",
        "reward-like output",
        f"step12_reward_like_{key}",
    )


#%% Step 13: Architecture-level statistics

architecture_stats = (
    summary_df
    .groupby(["architecture", "phenotype"])
    .agg(
        post_drift_mean=("post_drift", "mean"),
        post_drift_std=("post_drift", "std"),
        recovery_ratio_mean=("recovery_ratio_post_over_peri", "mean"),
        recovery_ratio_std=("recovery_ratio_post_over_peri", "std"),
        phase_transition_index_mean=("phase_transition_index", "mean"),
        phase_transition_index_std=("phase_transition_index", "std"),
        drift_auc_mean=("drift_auc", "mean"),
        drift_auc_std=("drift_auc", "std"),
        density_mean=("density", "mean"),
        spectral_radius_mean=("spectral_radius_estimate", "mean"),
    )
    .reset_index()
)

save_dataframe(architecture_stats, "architecture_level_statistics")



#%% Step 14: Perturbation-level statistics (maybe need more?)

perturbation_stats = (
    summary_df
    .groupby(["perturbation", "phenotype"])
    .agg(
        post_drift_mean=("post_drift", "mean"),
        post_drift_std=("post_drift", "std"),
        recovery_ratio_mean=("recovery_ratio_post_over_peri", "mean"),
        recovery_ratio_std=("recovery_ratio_post_over_peri", "std"),
        phase_transition_index_mean=("phase_transition_index", "mean"),
        phase_transition_index_std=("phase_transition_index", "std"),
        drift_auc_mean=("drift_auc", "mean"),
        drift_auc_std=("drift_auc", "std"),
    )
    .reset_index()
)

save_dataframe(perturbation_stats, "perturbation_level_statistics")


#%% Step 15: Resilience classification index

summary_df["resilience_index"] = (
    -summary_df["post_drift"]
    -summary_df["persistent_drift_post_minus_pre"]
    -summary_df["phase_transition_index"]
    -summary_df["recovery_ratio_post_over_peri"]
)

summary_df["resilience_rank_within_perturbation"] = (
    summary_df
    .groupby(["perturbation"])["resilience_index"]
    .rank(ascending=False)
)

save_dataframe(summary_df, "factorial_summary_statistics_with_resilience_index")

resilience_stats = (
    summary_df
    .groupby(["architecture", "phenotype"])
    .agg(
        resilience_index_mean=("resilience_index", "mean"),
        resilience_index_std=("resilience_index", "std"),
        post_drift_mean=("post_drift", "mean"),
        recovery_ratio_mean=("recovery_ratio_post_over_peri", "mean"),
    )
    .reset_index()
)

save_dataframe(resilience_stats, "resilience_index_statistics")

plot_group_bar(
    resilience_stats,
    x_col="architecture",
    y_col="resilience_index_mean",
    group_col="phenotype",
    title="Resilience index by architecture and phenotype",
    name="step15_resilience_index_by_architecture_phenotype",
)

#%% Step 16: Save manuscript-ready text summary as a (keeplike this?)

best_rows = summary_df.sort_values("resilience_index", ascending=False).head(20)
worst_rows = summary_df.sort_values("resilience_index", ascending=True).head(20)

best_path = STATISTICS_DIR / f"top_resilient_conditions_{RUN_STAMP}.csv"
worst_path = STATISTICS_DIR / f"top_vulnerable_conditions_{RUN_STAMP}.csv"

best_rows.to_csv(best_path, index=False)
worst_rows.to_csv(worst_path, index=False)

summary_text = f"""
Run stamp: {RUN_STAMP}

The experiment compared dense, sparse, modular, and small-world residual recurrent architectures in Nengo.

Each network contained explicit excitatory and inhibitory (EI) populations with an 80:20 E to I ratio.

Control, resilient, and vulnerable_depressive phenotypes were implemented as distinct dynamical regimes.

Stress progressively altered E/I balance through inhibition loss, recurrent gain increase, elevated neural variability, and altered recurrent leak dynamics.

Biologically inspired perturbation mechanisms included Gaussian noise injection, residual pathway degradation, synaptic gain amplification, heterogeneous synaptic time constants, partial neuronal silencing, and recurrent instability induction.

Primary outputs were state drift, post-stress recovery ratio, persistent post-stress deviation, E/I balance, freezing-like output, reward-like output, and phase transition index.

Best condition file:
{best_path}

Worst condition file:
{worst_path}
"""

text_path = RESULTS_DIR / f"run_summary_{RUN_STAMP}.txt"
text_path.write_text(summary_text)

print(summary_text)
print("Saved:", text_path)
print("Done.")


#%% Step 17: Save W_r and adjacency matrices for pre, peri, and post phases as compressed NPZ files

MATRIX_DIR = RESULTS_DIR / "matrices"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

def phase_mean(values, phase_name):
    s, e = phases[phase_name]
    values = np.asarray(values)
    values = align_length(values, TIME_STEPS)
    return float(np.mean(values[s:e]))

def save_wr_adjacency_npz(architecture, phenotype, perturbation, seed):
    dynamics = EIResidualResNetDynamics(
        architecture=architecture,
        phenotype=phenotype,
        perturbation=perturbation,
        seed=seed,
    )

    adjacency = dynamics.mask.astype(np.float32)

    for phase_name in ["pre", "peri", "post"]:
        mean_load = phase_mean(stress_load, phase_name)
        inhibition_eff = float(np.clip(
            1.0 - dynamics.inhibition_loss_strength * mean_load,
            0.05,
            2.0,
        ))
        recurrent_gain = float(dynamics.base_gain + dynamics.gain_increase_strength * mean_load)

        W_r = dynamics.effective_weight(inhibition_eff).astype(np.float32)
        W_r_scaled = (W_r * recurrent_gain).astype(np.float32)

        filename = (
            f"W_r_adjacency_"
            f"architecture-{architecture}_"
            f"phenotype-{phenotype}_"
            f"perturbation-{perturbation}_"
            f"phase-{phase_name}_"
            f"seed-{seed}_"
            f"{RUN_STAMP}.npz"
        )

        path = MATRIX_DIR / filename

        np.savez_compressed(
            path,
            W_r=W_r,
            W_r_scaled=W_r_scaled,
            adjacency=adjacency,
            architecture=architecture,
            phenotype=phenotype,
            perturbation=perturbation,
            phase=phase_name,
            seed=seed,
            run_stamp=RUN_STAMP,
            n_units=N_UNITS,
            n_exc=N_EXC,
            n_inh=N_INH,
            e_ratio=E_RATIO,
            mean_stress_load=mean_load,
            inhibition_eff=inhibition_eff,
            recurrent_gain=recurrent_gain,
            residual_scale=dynamics.residual_scale,
            active_connections=int(np.sum(adjacency)),
            total_connections=int(adjacency.size),
            density=float(np.sum(adjacency) / adjacency.size),
        )

        print("Saved:", path)

for seed_offset in range(N_SEEDS):
    seed = SEED + seed_offset

    for architecture in ARCHITECTURES:
        for phenotype in PHENOTYPES:
            for perturbation in PRIMARY_PERTURBATIONS:
                save_wr_adjacency_npz(
                    architecture=architecture,
                    phenotype=phenotype,
                    perturbation=perturbation,
                    seed=seed,
                )

print("Saved all W_r and adjacency matrices to:", MATRIX_DIR)
