Project status: Planned, on hold

# Background

Stress is a ubiquitous phenomenon that affects individuals across various contexts. Whether acute or chronic, positive or negative, physical or psychological, stress can serve as a motivating force or contribute to significant health impairments. Individuals who effectively manage stressful or even traumatic events and recover swiftly are considered stress resilient, whereas those who struggle to cope with stress are classified as vulnerable, also referred to as stress-susceptible individuals [1,2]. Stress resilience research aims to elucidate the factors and mechanisms that foster resilience, and to develop preventative interventions for individuals at risk of stress-related dysfunctions and mental disorders [3,4]. This research employs a multidisciplinary approach, integrating neurobiological analyses, physiological and psychological assessments, as well as longitudinal cohort studies to examine resilience trajectories over time [2,3,5,6].
While empirical and observational methods have traditionally shaped resilience research, theoretical and computational approaches offer a complementary and mechanistic framework for 
investigation [7]. Importantly, acute and chronic stress differ in their effects on the brain, with consequences for neural plasticity and functional stability [8, 9], which are shaped by individual differences in stress-resilience versus vulnerability [10]. Brain network simulation models allow controlled perturbations to probe resilience, maladaptive responses, and recovery dynamics over time, and provide a pathway toward real-time, embodied implementations that can connect neural dynamics to behavior.

The primary objective is to construct resilient and vulnerable computational brain network models, and to examine how controlled perturbations of varying intensity and duration affect their dynamics and stability before, during, and after stress, enabling comparisons of acute and chronic stress to study stress resilience.

# Project for the Nengo Summer School


# little prep 

# Biological Recurrent Neural Networks for Investigating Stress-Induced Vulnerability and Resilience

IMPORTANT NOTE: all networks that will be used in Nengo and have been used 2025 were recurrent networks, also in case I use SPAUN. 
I added the ei balance nearly directly in 2025 because this makes the networks more biological + I also want to ensure neuronal heterogenity for the populations.
Other biological features: noise, non-linearities

## Motivation

A major limitation of conventional machine learning (ML) neural networks is that they are optimized primarily for predictive performance rather than biological interpretability. Standard deep learning architectures rely on backpropagation and distributed internal representations in which individual nodes typically lack direct biological meaning. Consequently, it is difficult to relate changes in network activity or connectivity to experimentally observed neural mechanisms underlying stress-related disorders.

The objective of this project is therefore to develop biologically informed recurrent neural network models that enable direct investigation of neural dynamics, synaptic connectivity, and behavioral outcomes under stress. By incorporating biologically meaningful neuron populations and connectivity constraints, the model should allow examination of how stress-induced circuit alterations translate into changes in behavior.

A particular focus will be placed on generating and comparing two classes of networks:

1. **Resilient networks**, which maintain performance despite stress-related perturbations.
2. **Vulnerable networks**, which exhibit maladaptive changes resembling stress-related psychopathology.

This framework aims to establish an initial link between neural circuit dynamics and behavioral phenotypes.

---

# prep for first presentation

#reminder 

# SPAUN -> possible future work 

is a recurrent NW -> use this in later stage for the mouse to compare task performance in my models vs SPAUN? useful to look at least for stressor which NW metrics change here under stress?

# The Brain as a Recurrent Dynamical System

The mammalian brain can be conceptualized as a recurrent dynamical system in which neural populations continuously interact through feedback connections. Unlike feedforward architectures, recurrent networks naturally capture temporal dependencies, memory processes, and ongoing state transitions that are fundamental to cognition and behavior.

Several neural network architectures will be reviewed as conceptual background:

* Convolutional Neural Networks (CNNs)
* Recurrent Neural Networks (RNNs) -> EI-RNNS for mPFC, Hippocampus together with Amygdala can be discussed -> focus on dynamics and Network architecture
* Residual Networks (ResNets)
* Long Short-Term Memory Networks (LSTMs) -> reuse but transfer work from 2025 (preparation for NSS)
* Other recurrent architectures discussed in previous work 2025 (preparation for NSS)

While these architectures provide useful computational principles, the primary focus of this project will be on network metrics (transitivity, clustering...), network topology, and biologically constrained recurrent networks over time (dynamics!). I further would like to have a network of 500 neurons to start (reason: easier to check the network metrics).

---



# Biologically Inspired Network Architecture

The proposed framework will employ biologically plausible neural populations rather than abstract computational units. Implementation may utilize platforms such as Nengo, which support heterogeneous neuron models and biologically meaningful circuit organization.

### Cortical Network Structure

A key design principle will be preservation of the approximate cortical excitatory-to-inhibitory (E/I) ratio:

* 80% excitatory neurons
* 20% inhibitory neurons

This excitatory-inhibitory balance is a fundamental property of cortical circuits and is strongly implicated in stress-related dysfunction.

### Initial Brain Regions

The first model will focus on one of two candidate modules:

* **Medial Prefrontal Cortex (mPFC)**
* **Hippocampus (HIPP)**

* Later stage will then focus on the interaction of mPFC-Il,PL-HIPP-AMY (Amygdala) for the rodent in the arena (discuss the populations related to cognitive map, place, splitter, grid etc cells)

The medial prefrontal cortex is particularly attractive because extensive literature links stress exposure to alterations in mPFC connectivity, inhibitory control, and behavioral regulation.

---

# Stress-Induced Network Perturbations

The central hypothesis is that stress modifies neural connectivity and E/I balance, producing distinct dynamical states associated with resilience or vulnerability.

The model will therefore incorporate domain knowledge from:

* Depression literature
* Post-traumatic stress disorder (PTSD) literature
* Rodent stress models

Potential manipulations include:

* Altered E/I balance in the papers (use it in a general way or in a specific way?)
* Synaptic weight modifications
* Changes in inhibitory control
* Region-specific connectivity disruptions

Ultimately, the goal is to generate network states that resemble resilient and depressive/vulnerable brain phenotypes.

---

# Learning Framework

The network will initially be trained using supervised optimization.

Conceptually:

Input → Randomly initialized connection weights → Biologically meaningful neurons → Neural activity decoding → Output

Training will adjust synaptic weights to minimize prediction error, for example:

(x-\hat{x})^2 <- discuss also the other optimization ideas!

where x represents the target output and x̂ the network prediction.

Unlike conventional deep learning approaches, the focus is not solely on performance but also on understanding how learning alters internal network organization.

---

# Validation Through Direction-Based Tasks

As an initial validation step, the network will be trained on simple sensorimotor discrimination tasks.

Potential inputs include:

* Tensors
* Images
* Rotated images
* Directional stimuli

Example behavioral outputs:

* Up vs. down
* Left vs. right
* Direction of movement using Up, down, right, left
* Rotation direction

These tasks are intentionally simple to facilitate interpretation of emerging neural representations.

The primary objective of this stage is to establish a direct relationship between:

Neural activity patterns → Network connectivity → Behavioral performance

within a single biologically informed brain network.

---

# Importance of Network Dynamics

A major advantage of this framework is the ability to examine neural dynamics continuously throughout learning and stress exposure.

Rather than analyzing connectivity only before and after training, it becomes possible to extract:

* Weight matrices at any time point
* Population activity dynamics
* Changes in E/I balance
* Evolution of network attractor states

This enables investigation of how stress-related perturbations reshape circuit organization over time.

---

# Future Extensions

After validation with simple directional tasks, more complex cognitive and behavioral paradigms can be introduced.

Potential future tasks include:

* Motor control tasks
* Navigation paradigms
* Virtual rodent arena experiments
* Working memory tasks
* Decision-making tasks

These extensions would permit progressively richer comparisons between resilient and vulnerable network phenotypes.

---

# Expected Contribution

This project aims to bridge computational neuroscience and machine learning by constructing biologically interpretable recurrent neural networks that model stress-induced changes in neural circuitry.

The long-term goal is to identify how specific alterations in neural connectivity and excitatory-inhibitory balance produce behavioral signatures associated with resilience, depression, and PTSD. By explicitly modeling biologically meaningful neuron populations and circuit dynamics, the framework may provide mechanistic insights that are difficult to obtain from conventional deep learning models.










--- 

move to git

import nengo
import numpy as np

model = nengo.Network()

#reminder for two neurons
# Alice and Bob neuron 
# both are sensitive to one position: x position of the eye

#for 40 neurons

with model:
    ens = nengo.Ensemble(n_neurons=40, dimensions=1)
    
    
    stim = nengo.Node([0])
    nengo.Connection(stim, ens)
    
    output = nengo.Node(None, size_in=1)
    
    def square(x):
        return x*x
        
    nengo.Connection(ens, output, function=square)
    
