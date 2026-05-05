# Overall Analysis Plan 

This "plan" resulted from the failures with the Ca-Imafing data, see notebook

#TODO seems here the hub part + loops, eg transitivity loop is missing (i had this in mind and it is in the latex, but not here - did I forgot it in DEC???)... :/ I need to check the LaTeX!!! 

There are several points that, I think, were not good practice. Most notably the __not normalizing for number of nodes__.

The goal of this workflow is to understand how resilience training alters the **structure and dynamics of recurrent neural networks**, with a particular focus on inhibitory control and network stability.

I will model brain networks with Nengo. Before analyzing the Interaction of 2NODE (mPFC-AMY) or 3NODE(mPFC-HIPP-AMY), I want to take a look on the 1NODE == mPFC (with IL, PL projections, recurrence (ResNet, RNN basic structure, maybe CNN?, LSTM or GRU components), ei balance, resulting in a weihgted, directed graph like neural network). Recurrence and ei balance will make these networks "biologically meaningful". I want to see, if I can identify nw metrics (see CAI code), hubs, loops... any patterns that explain stress resilience.

I want to apply what I can learn from artificial networks to the network analysis of real data. Either by using real data to improve the models or by finding ways to better analyze real data. 

I structure the analysis from **basic validation → structural analysis → weighted organization → dynamical implications**. After Nengo, I think I will adapt this workflow anyway, so this is for now just a possible guideline in case I will have enough time already during the Nengo Summer School to analyze the 1NODE = mPFC model.

Resilient vs vulnerable models (for Nengo just one, but for later validation the models can have different architectures, different start configurations in the context of ei balance and also stress will differ -> see diff. stressors that are "biologically meaningful") -> for Nengo best: depressive model (or PTSD?), increased stress over time.

---

## 1. Quality Control & Weight Distribution

Before performing any network analysis, I first "verify" that the networks are well-formed and comparable across conditions.

#TODO discuss this, what is a good way to do QC, I popose:

### What I check:

- Distribution of weights \( W_{ij} \)
- Mean and variance of weights
- Balance of positive (excitatory) vs negative (inhibitory) weights
- Presence of many near-zero weights (sparsity from training)
- use log scale for balance in the network

Note: I will report the mean of log10(E/I), which corresponds to the geometric mean of the E/I ratio, rather than using an arithmetic average of ratios. This choice is motivated by the fact that E/I is inherently a multiplicative quantity: doubling excitation relative to inhibition (E/I = 2) and halving it (E/I = 0.5) represent symmetric deviations from balance. The geometric mean preserves this symmetry, whereas the arithmetic mean would bias the result toward larger ratios. The log transformation makes this symmetry explicit, such that log10(E/I) = 0 corresponds to perfect balance (E = I), positive values indicate excitation dominance, and negative values indicate inhibition dominance.

### Typical plots:

- Histogram of weights
- Log-scale magnitude distribution
- Fraction of weights close to zero

### Goal:

I want to ensure that any observed differences are not due to trivial artifacts such as:
- global scaling differences
- degenerate weight distributions
- failed or unstable training runs

Note: I truely believe I should have normalized the number of nodes + check "redundancy" as metric (+ density here?)

---

## 2. Thresholding Strategy (Graph Construction)

Since the networks are either sparse, (maybe also a inbetween, 50%) or nearly fully connected, I define a principled way to extract meaningful graph structure.

### Approach:

- I will use **quantile-based thresholding**
- For example:
  - keep top 10%, 20%, or 30%, 50% strongest connections 

Note: Ca-Imaging networks needed a threshold of 0.2. This is why I start low, compare results and check wether the results are robust
#TODO plot the networks like for the CAI, like this I can see if there are any edges, this was really a trouble as too high thresholds resulted in nearly only single nodes (high subclusters, by the way)

### Validation:

- I perform a **threshold sensitivity analysis**
- I verify that my results are stable across thresholds

### Goal:

I want to avoid arbitrary threshold choices and ensure that structural conclusions are robust.

---

## 3. Block-wise Connectivity Analysis

I partition the network into canonical sub-blocks:

- E→E
- I→E
- E→I
- I→I

### What I compute:

- Mean weight magnitude
- Edge density (after thresholding)
- Statistical comparisons between conditions

### Goal:

I want to identify **where connectivity changes occur**, especially:

- whether inhibition onto excitatory neurons decreases
- whether inhibitory self-connectivity increases

---

## 4. E/I Balance Analysis

I quantify excitation vs inhibition at the level of individual neurons

### Definition:

$$
E_i = \sum_{j \in E} W_{ij}, \quad
I_i = \sum_{j \in I} W_{ij}
$$

$$
R_i = \frac{|E_i|}{|I_i|}
$$

### What I analyze:

- Distribution of \( \log_{10}(R_i) \)
- Network-level mean E/I balance

### Goal:

I want to test whether resilience shifts the network toward **stronger inhibitory dominance**.

---

## 5. Weighted Network Analysis

I move beyond binary graphs and analyze the full weight structure.

### Components:

#### 5.1 Block-wise weighted strength
- How total synaptic weight is distributed across E/I blocks

#### 5.2 E/I balance (weighted)
- Magnitude of excitatory vs inhibitory input

#### 5.3 Heterogeneity
- Variance, tail behavior (what happens at the extremes of a distribution-chaos like?), and presence of hubs

#### 5.4 Node-level analysis
- Differences at the level of individual neurons (E vs I populations)

### Key question:

Is total synaptic strength changed, or is it redistributed?
or: 
Does resilience alter the global magnitude of synaptic weights, or does it reorganize their distribution across specific connectivity motifs and neuron populations?
Is synaptic strength globally scaled, or conserved and selectively reallocated across excitatory and inhibitory pathways?


---

## 6. Directed Graph Metrics

I construct directed graphs and compute:

- In-degree / out-degree
- Centrality measures (PageRank, Katz, etc.)
- Connectivity structure (weak/strong components)
- Path-based metrics

### Important distinction:

- Unweighted graphs → topology
- Weighted graphs → influence

### Goal:

I want to characterize how the **structure of information flow** changes.

---

## 7. Redundancy & Subnetwork Structure

I analyze robustness-related structure, focusing on:

- Redundancy in I→E connections
- Changes in I→I loops
- Alternative inhibitory pathways

### Goal:

I want to test whether resilience increases **redundant inhibitory control mechanisms**.

---

## 8. Spectral & Dynamical Analysis

I analyze the dynamical regime of the network.

### Metrics:

- Spectral radius \( \rho(W) \)
- Eigenvalue distribution
- Distance to instability (edge of chaos)

### Goal:

I want to understand how structural changes affect:

- stability
- sensitivity to perturbations
- dynamical robustness

---

## 9. Edge of Chaos / Stability Regime

I interpret spectral results in dynamical terms:

- Subcritical (stable)
- Critical (edge of chaos)
- Supercritical (unstable)

### Hypothesis:

I expect resilient networks to operate in a regime with:

- stronger inhibitory stabilization
- controlled dynamics under perturbation

---

## 10. Integration & Interpretation

Finally, I combine all results.... hopefully :D

### Key dimensions:

- Global vs local changes
- Weight conservation vs redistribution
- Structural vs dynamical effects

### Core question:

- Does resilience emerge through increased inhibition, or through a reorganization of inhibitory control?
- Vulnerable = depressive/PTSD -> Do the findings in the artificial NN match with findings in real data? (chronic strss)
#TODO I urgently have to read papers!

---

## Summary of Pipeline so far (2025DEC22)

1. Quality control (weights)
2. Thresholding strategy
3. Block-wise structure
4. E/I balance
5. Weighted organization
6. Directed graph structure
7. Redundancy
8. Spectral analysis
9. Dynamical regime
10. Interpretation

Last coding: 2025DEC22, proposal for 2026 wrtitten, chat with Janina -> critical: is the ResNet +-LSTM +ei balance enough? -> because of this concern, I added using the BG model + SPAUN

---

### MANY Questions for (or before?) Nengo 2026

- What is the best way to do the "benchmarking" (with/without LSTM or the architectures first? Directly with ei balance?)
- Use a connectivity matrix? similar approach like for CAI data
- Tone window style: use snapshots of the network before and after stress, use a fixed time schedule for this!?
- define a stress paradigm like an actual experiment
- mimic the CAFC (tones at later stages) a good idea?
- very later: get behavior, freezing (outcome) #discuss
- Should the benchmarking include both the original RNNs and LSTM-based models, or should I first establish the analysis pipeline differently?
- Use the resilient == CONTROL in the experiment as a main baseline?
- Which metrics should define the primary benchmark: task performance, perturbation robustness, E/I balance, graph structure, or dynamical stability?
- Use real data? Or better just artificial first (for mouse trajectories I could use real data, see top Nenngo ideas file)
- Main sample size = 500 (ensemble)?
- Should thresholded graph analysis be used as a structural benchmark, while weighted analysis is used as a mechanistic follow-up?
- Are global metrics like mean degree and total strength informative, or mainly useful as controls showing that the total connectivity budget is preserved?
- Should I start with global network metrics or directly focus on block-specific metrics such as I→E and I→I?
- Should redundancy be defined through shared input overlap, alternative paths, or robustness to edge removal?
- directed:in/out-degree, PageRank, Katz centrality, betweenness, or strongly connected components? (see interdisciplinary paper resilience, robustness)
- Should signed weights be analyzed separately from absolute weights, given that excitatory and inhibitory edges hav
- Which results are purely descriptive, and which can support a mechanistic claim?
- Does the observed shift toward stronger inhibition explain robustness directly, or is additional perturbation-based validation needed?
- Should resilience-trained networks be compared against shuffled controls to show that the observed block structure is meaningful?
#reminder CAI shuffling!
- For adding more nodes: What is the best way to study the interactions of the brain areas in an appropriate and clean way?
- For just AMY: keep the AMY simple (just one value at time for the start)
- Neuromodulation see LaTeX top Nengo ideas! Multipling edges after the initial analysis of the ei balance -> do i need this very early to make it enough "biologically meaningful"
- best way to analyze loops, hubs

---

## Expected Outcome after Nengo (Q3 or Q4 2026)

I expect to show that:

- Global connectivity is largely preserved
- Inhibitory structure is reorganized
- Resilience emerges through **targeted redistribution**, not global scaling
- These structural changes shift the network toward a more stable dynamical regime
- After Gating, 3-5NODE: What impact do the other brain regions have? 
