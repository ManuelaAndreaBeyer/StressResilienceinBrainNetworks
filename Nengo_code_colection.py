import nengo

model = nengo.Network()

with model:
    stim = nengo.Node([0)]
    a = nengo.Ensemble(n_neurons=50, dimension=1)
    nengo.Connection(stim,a)
    
    b = nengo.Ensemble(n_neurons=100, dimesion=1)
    

    nengo.Connection(a,b, synapse=0.2)

#-----------------------------
