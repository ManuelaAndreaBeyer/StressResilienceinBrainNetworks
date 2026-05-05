# Overall Analysis Plan 

Use this as a guideline to analyze graph-like neural networks that use recurrence, ei balance and neuromodulation to make the networks "biologically meaningful"

## 1. Quality Control & Weight Distribution

tba

## 2. Thresholding Strategy (Graph Construction)

tba

## 3. Block-wise Connectivity Analysis

- E→E
- I→E
- E→I
- I→I

## 4. E/I Balance Analysis

tba

## 5. Weighted Network Analysis

tba

## 6. Directed Graph Metrics

tba

## 7. Redundancy & Subnetwork Structure

tba


## 8. Spectral & Dynamical Analysis

tba

## 9. Edge of Chaos / Stability Regime

tba

## 10. Integration & Interpretation

tba

### Key dimensions:

- Global vs local changes
- Weight conservation vs redistribution
- Structural vs dynamical effects

## Summary of preliminary pipeline 
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
