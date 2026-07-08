#todo discuss this and how it may be transferred to nengo
#commit discuss how ei rnn can be transferred to nengo, adding functional companents and having a depressive model to compare it to the resilient (well performing enough?)

#simple backbones 

# not sure if i can use this anymore


# IMPORTANT: these implementations are not biological enough, so they have to be adjusted and improved as they are too "obvious"

# discuss what may make these starting architectures more concrete and specific, only a delay task is not enough

# feedback: scientific gap! Many EI-RNN paper deal with mPFC-like, stress and I will probably work on one of my other ideas that do not use the mPFC module as a single component
---
"""


Cell type specific populations as small networks, use spiking neural networks (LIF) first, explore continuous time like 2025 later

Start of Nengo: Explore original idea for this summer school: 
crtitical: proposal is superficial

depression like phenotype:
| Region |         Cell type | Chronic stress effect                 | Model implementation                |
| ------ | ----------------: | ------------------------------------- | ----------------------------------- |
| PL     | Pyramidal neurons | increased I/E via stronger inhibition | increase inhibitory gain onto PL-PN |
| IL     | Pyramidal neurons | increased I/E via weaker excitation   | decrease excitatory gain onto IL-PN |
| PL/IL  |   PV interneurons | largely unaffected                    | keep PV modulation near 1           |
Paper I decided to use for the mpfc module: https://pubmed.ncbi.nlm.nih.gov/39147579/


idea: transfer EI-RNNs representing mPFC, HIPP AMY modules to Nengo, add neuromodulation?
I thought, we have to start with one module and add functional components here, but we found a better approach :)

-> still open: discuss how the "old" idea can be used to study stress resilience. static networks after training

"""
#TODO discuss and sort out later, just backup early implementation using ai
