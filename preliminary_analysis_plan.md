# Overall Analysis Plan for network analysis - Nengo Summer School 2026 preparation

Usage: 

This as a workflow to analyze different graph-like neural networks that use recurrence, E/I balance and optionally neuromodulation to make the networks "biologically meaningful". 

---

For me, the brain is a network and I would love to analyze the constructed netorks alike, starting with a network model for the mPFC.

The long-term goal is to investigate whether biologically inspired network architectures can provide mechanistic insights into stress resilience and adaptive network dynamics. My particular interest is in comparing different network architectures and metrics and see how they change under acute vs chronic stress. I am also interested in detecting transitions, that already came up in implementation and I would welcome the chance to discuss very preliminary findings from 2025 and how the implementations can be improved.

---

Most time from January 2025 to December 2025: I plenned to have 2 or more steps, one version to discuss:

Steps -Status quo 2026 MAY - reccurence and ei balance in graph like brain netowrks


validation step that uses one network - I haven't decided if it is really the mPFC with IL, PL projections from the RNN and ResNet OR another cortical network (nothing implemented here, so I currently think I will do eihther mPFC or directly use mPFC-HIPP) - construction still with ei like in all architectures for paper: compare different architectures
-> NW analysis, 4 ei blocks for 1NODE


Add the Amygdala 2NODE Analysis


Fear vs Reward Condotioning (arena idea), could be also implemented in a robot; tensor, shock, tone -> taks: alternating water reward -> behavioral response
I think I will need here 2-3, max. 5NODE


The above model will have mPFC (Il,PL)-HIPP-AMY nodes - this was the idea for the Generalized model and I thought I can link it and put it on one paper, but at the moment I am no longer sure about it...
This step now will include neuromodulation (see how to in Latex cortisol, HPA achsis idea or use any ei relevant)



5NODE? Add more brain areas NAcD1/2, Hypothalamus.. resilience relevant (see documents, papers on zotero), Basal Ganglia, SPAUN if not already done



More disease models in collaborations?

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
