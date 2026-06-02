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
#---

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

    #----

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
        #return (tau_synapse/tau_desired)*u
    nengo.Connection(a,b, synapse=tau_synapse, function=tau_synapse)
    
    def func(x):
        #recurrent function
        return -(tau_synapse/tau_desired)x*x
        
    nengo.Connection(b,b,synapse=tau_synapse)
    
    control=nengo.Enseble

    #---

    import nengo
import numpy as np

#da/dt=omega*b
#db/dt=omega*a

omega = 2*np.pi

model = nengo.Network()

with model:
    stim = nengo.Node([0])
    ens = nengo.Ensemble(n_neurons=2000, dimensions=2, radius=2)
    #nengo.Connection(stim,a)
    
    
    
    tau_synapse=0.1
    def recurrent(x):
        a,b=x
        return ((omega*tau_synapse*b+a), (-omega*tau_synapse*a+b))
    
    
    b = nengo.Connection(ens,ens,synapse=tau_synapse, function=recurrent)
    
    



#-----------------------------


import numpy as np
import nengo
import scipy.special



def get_weights_for_delays(q, r):
    # compute the weights needed to extract the value at time r
    # from the network (r=0 is right now, r=1 is theta seconds ago)
    r = np.asarray(r)
    m = np.asarray([scipy.special.legendre(i)(2*r - 1) for i in range(q)])
    return m.reshape(q, -1).T


def create_AB(q, theta):
    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    for i in range(q):
        B[i] = (-1.)**i * (2*i+1)
        for j in range(q):
            A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
    return A / theta, B / theta
    
    
    
model = nengo.Network()

with model:
    stim=nengo.Node(0)
    
    theta=0.5
    
    q=6
    
    A,B=create_AB(q,theta)
    #print(B)
    
    tau_synapse=0.1
    
    lmu=nengo.Ensemble(n_neurons=1000,dimensions=q)
    nengo.Connection(stim, lmu,transform=tau_synapse)
    nengo.Connection(lmu,lmu, transform=A*tau_synapse+np.idendity(q),synapse=tau_synapse)
