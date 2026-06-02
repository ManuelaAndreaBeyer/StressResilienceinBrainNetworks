import nengo

model = nengo.Network()

with model:
    stim = nengo.Node([0)]
    a = nengo.Ensemble(n_neurons=50, dimension=1)
    nengo.Connection(stim,a)
    
    b = nengo.Ensemble(n_neurons=100, dimesion=1)
    

    nengo.Connection(a,b, synapse=0.2)

#-----------------------------

import nengo

model = nengo.Network()


with model:
    stim = nengo.Node([0])
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim,a)
    
    
    #dx/dt=[1/tau_desired)*(u-x)
    
    #dx/dt=[1/tau_desired)*u + (-1/tau_desired)*x
    
    #forward tau_synapse*g(u)
    #recurrent tau_synapse*f(x)+x
    
    
    
    tau_synapse=0.02
    
    tau_desired=0.5
    
    
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    
    def forward(u):
        #input function
        return (tau_synapse/tau_desired)*u
    

    nengo.Connection(a,b, synapse=tau_synapse, function=forward)
    
    def func(x):
        #recurrent function
        return -(tau_synapse/tau_desired)x*x
        
    nengo.Connection(b,b, function=recurrent,synapse=tau_synapse)



#-----------------------------
