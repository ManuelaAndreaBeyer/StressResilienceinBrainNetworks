Project status: Planned, on hold

# Background

Stress is a ubiquitous phenomenon that affects individuals across various contexts. Whether acute or chronic, positive or negative, physical or psychological, stress can serve as a motivating force or contribute to significant health impairments. Individuals who effectively manage stressful or even traumatic events and recover swiftly are considered stress resilient, whereas those who struggle to cope with stress are classified as vulnerable, also referred to as stress-susceptible individuals [1,2]. Stress resilience research aims to elucidate the factors and mechanisms that foster resilience, and to develop preventative interventions for individuals at risk of stress-related dysfunctions and mental disorders [3,4]. This research employs a multidisciplinary approach, integrating neurobiological analyses, physiological and psychological assessments, as well as longitudinal cohort studies to examine resilience trajectories over time [2,3,5,6].
While empirical and observational methods have traditionally shaped resilience research, theoretical and computational approaches offer a complementary and mechanistic framework for 
investigation [7]. Importantly, acute and chronic stress differ in their effects on the brain, with consequences for neural plasticity and functional stability [8, 9], which are shaped by individual differences in stress-resilience versus vulnerability [10]. Brain network simulation models allow controlled perturbations to probe resilience, maladaptive responses, and recovery dynamics over time, and provide a pathway toward real-time, embodied implementations that can connect neural dynamics to behavior.

The primary objective is to construct resilient and vulnerable computational brain network models, and to examine how controlled perturbations of varying intensity and duration affect their dynamics and stability before, during, and after stress, enabling comparisons of acute and chronic stress to study stress resilience.

# Project for the Nengo Summer School

#TODO enter a clean version when finished

depression like phenotype:
| Region |         Cell type | Chronic stress effect                 | Model implementation                |
| ------ | ----------------: | ------------------------------------- | ----------------------------------- |
| PL     | Pyramidal neurons | increased I/E via stronger inhibition | increase inhibitory gain onto PL-PN |
| IL     | Pyramidal neurons | increased I/E via weaker excitation   | decrease excitatory gain onto IL-PN |
| PL/IL  |   PV interneurons | largely unaffected                    | keep PV modulation near 1           |
link to paper that was considered early before summer school and propably offers potential for discussion: https://pubmed.ncbi.nlm.nih.gov/39147579/

Cell type specific populations as small networks, use spiking neural networks (LIF) first, explore continuous time like 2025 later

Start of Nengo: Explore original idea for this summer school: 
crtitical: proposal is superficial

idea: transfer EI-RNNs representing mPFC, HIPP AMY modules to Nengo, add neuromodulation?
I thought, we have to start with one module and add functional components here, but we found a better approach :)

-> still open: discuss how the "old" idea can be used to study stress resilience. static networks after training
