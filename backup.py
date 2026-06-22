
#simple backbones 
---


import torch
import torch.nn as nn
import torch.nn.functional as F


class DaleEIProjection(nn.Module):
    """
    EI recurrent matrix with Dale constraint:
    columns = presynaptic neurons.
    Excitatory columns >= 0, inhibitory columns <= 0.
    """
    def __init__(self, n, p_exc=0.8, g=1.0):
        super().__init__()
        self.n = n
        n_exc = int(p_exc * n)
        sign = torch.ones(n)
        sign[n_exc:] = -1.0
        self.register_buffer("dale_sign", sign.view(1, n))  # column signs

        self.raw_w = nn.Parameter(0.1 * torch.randn(n, n))
        self.g = g

    def forward(self):
        w_pos = F.softplus(self.raw_w)
        w = w_pos * self.dale_sign
        w = w / (self.n ** 0.5) * self.g
        return w


class RateEIRNN(nn.Module):
    """
    Rate-based EI-RNN:
    x[t+1] = (1-alpha)x[t] + alpha * tanh(x[t]Wrec + u[t]Win + b)
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 il_dim=None, pl_dim=None, p_exc=0.8, dt=1.0, tau=20.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = dt / tau

        self.win = nn.Linear(input_dim, hidden_dim, bias=False)
        self.rec = DaleEIProjection(hidden_dim, p_exc=p_exc)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # IL / PL projections
        self.il_proj = nn.Linear(il_dim, hidden_dim, bias=False) if il_dim else None
        self.pl_proj = nn.Linear(pl_dim, hidden_dim, bias=False) if pl_dim else None

        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, u, il=None, pl=None):
        """
        u:  [batch, time, input_dim]
        il: [batch, time, il_dim] or None
        pl: [batch, time, pl_dim] or None
        """
        batch, time, _ = u.shape
        x = torch.zeros(batch, self.hidden_dim, device=u.device)
        Wrec = self.rec()

        states = []
        outputs = []

        for t in range(time):
            drive = self.win(u[:, t]) + x @ Wrec + self.bias

            if self.il_proj is not None and il is not None:
                drive = drive + self.il_proj(il[:, t])

            if self.pl_proj is not None and pl is not None:
                drive = drive + self.pl_proj(pl[:, t])

            x = (1 - self.alpha) * x + self.alpha * torch.tanh(drive)
            y = self.readout(x)

            states.append(x)
            outputs.append(y)

        return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


---


class SpikeFn(torch.autograd.Function):
    """
    Surrogate-gradient spike function.
    Forward: hard threshold.
    Backward: smooth triangular surrogate.
    """
    @staticmethod
    def forward(ctx, v_minus_th):
        ctx.save_for_backward(v_minus_th)
        return (v_minus_th > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        surrogate_grad = torch.clamp(1.0 - x.abs(), min=0.0)
        return grad_output * surrogate_grad


spike_fn = SpikeFn.apply


class SpikingEIRNN(nn.Module):
    """
    Spiking EI-RNN with LIF neurons.
    Dale constraint is applied to recurrent synapses.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 il_dim=None, pl_dim=None, p_exc=0.8,
                 dt=1.0, tau_mem=20.0, v_th=1.0, reset=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = torch.exp(torch.tensor(-dt / tau_mem))
        self.v_th = v_th
        self.reset = reset

        self.win = nn.Linear(input_dim, hidden_dim, bias=False)
        self.rec = DaleEIProjection(hidden_dim, p_exc=p_exc)

        self.il_proj = nn.Linear(il_dim, hidden_dim, bias=False) if il_dim else None
        self.pl_proj = nn.Linear(pl_dim, hidden_dim, bias=False) if pl_dim else None

        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, u, il=None, pl=None):
        """
        u:  [batch, time, input_dim]
        il: [batch, time, il_dim] or None
        pl: [batch, time, pl_dim] or None
        """
        batch, time, _ = u.shape
        v = torch.zeros(batch, self.hidden_dim, device=u.device)
        s = torch.zeros(batch, self.hidden_dim, device=u.device)
        Wrec = self.rec()

        spikes = []
        outputs = []

        for t in range(time):
            current = self.win(u[:, t]) + s @ Wrec

            if self.il_proj is not None and il is not None:
                current = current + self.il_proj(il[:, t])

            if self.pl_proj is not None and pl is not None:
                current = current + self.pl_proj(pl[:, t])

            v = self.beta.to(u.device) * v + current
            s = spike_fn(v - self.v_th)
            v = v * (1.0 - s) + self.reset * s

            y = self.readout(s)

            spikes.append(s)
            outputs.append(y)

        return torch.stack(outputs, dim=1), torch.stack(spikes, dim=1)
