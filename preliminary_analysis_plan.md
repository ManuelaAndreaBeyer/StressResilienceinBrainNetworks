# Overall Analysis Plan - Nengo Summer School 2026 preparation

Use this as a workflow to analyze different graph-like neural networks that use recurrence, ei balance and neuromodulation to make the networks "biologically meaningful". For me, the brain is a network and I would love to analyze the constructed netorks alike.

A validation in torch assessing the applicability of brain network analysis to resilience research has been initiated using multiple stress paradigms (acute vs. chronic stress) and network architectures. Preliminary findings indicate chronic stress–associated alterations in network organization and dynamics.

Old torch implementation #upload here or keep this repo nengo-only? or for later paper in supplements? 

Important note: I used Recurrence on a high level and not for modeling single neurons. 

I would welcome the chance to learn about which specific machine learning-like computations can help to make networks more plausible.

This workflow is intended to analyze graph-based neural network models incorporating recurrence, excitatory/inhibitory (E/I) balance, and neuromodulatory mechanisms in order to improve biological plausibility.

The long-term goal is to investigate whether biologically inspired network architectures can provide mechanistic insights into stress resilience and adaptive network dynamics. My particular interest is in comparing different __network architecture and network metrics__ and how they change under stress. I also interested in possible transitions, that already came up and would welcome the chance to discuss this.

This line of work originated in the context of the initial application for the Nengo Summer School 2025. 


--- 

All implementations of cortical networks ( I am not sure which I will finally choose after the Summer School) can be analyzed with the following pipeline:
This pipeline resulted from failures I had with a pipeline implementation 2024, it was already possible to use it to analyze networks with it :)

--- 

# Pipeline 

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
