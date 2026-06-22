#todo discuss this and how it may be transferred to nengo
#commit discuss how ei rnn can be transferred to nengo, adding functional companents and having a depressive model to compare it to the resilient (well performing enough?)

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


---

# ============================================================
# Neuromodulated Dale-constrained EI-RNN
# ============================================================

class NeuromodulatedDaleEIRNN(nn.Module):
    """
    Rate-based EI-RNN with:

    - Dale constraint
    - excitatory / inhibitory populations
    - PL / IL regional split
    - chronic-stress-like neuromodulation

    Biological mapping:
        PL stress effect:
            increased inhibition onto PL pyramidal units

        IL stress effect:
            decreased excitation onto IL pyramidal units

        PV-like inhibitory units:
            no direct stress modulation
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=200,
        output_dim=1,
        p_exc=0.8,
        pl_fraction=0.5,
        dt=1.0,
        tau=20.0,
        rec_gain=1.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alpha = dt / tau
        self.rec_gain = rec_gain

        # ---------- region masks ----------
        n_pl = int(hidden_dim * pl_fraction)

        is_pl = torch.zeros(hidden_dim).bool()
        is_il = torch.zeros(hidden_dim).bool()

        is_pl[:n_pl] = True
        is_il[n_pl:] = True

        self.register_buffer("is_pl", is_pl)
        self.register_buffer("is_il", is_il)

        # ---------- cell-type masks ----------
        n_exc = int(hidden_dim * p_exc)

        is_exc = torch.zeros(hidden_dim).bool()
        is_inh = torch.zeros(hidden_dim).bool()

        is_exc[:n_exc] = True
        is_inh[n_exc:] = True

        self.register_buffer("is_exc", is_exc)
        self.register_buffer("is_inh", is_inh)

        # Dale sign: columns are presynaptic neurons
        dale_sign = torch.ones(hidden_dim)
        dale_sign[is_inh] = -1.0
        self.register_buffer("dale_sign", dale_sign.view(1, hidden_dim))

        # ---------- parameters ----------
        self.w_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_raw = nn.Parameter(0.05 * torch.randn(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.readout = nn.Linear(hidden_dim, output_dim)

        # Stress modulation strengths
        self.pl_inh_gain_strength = nn.Parameter(torch.tensor(0.5))
        self.il_exc_supp_strength = nn.Parameter(torch.tensor(0.5))

        # Optional global neuromodulatory gain
        self.global_gain = nn.Parameter(torch.tensor(0.5))

    def dale_weight(self):
        W = F.softplus(self.w_raw)
        W = W * self.dale_sign
        W = W / math.sqrt(self.hidden_dim)
        W = W * self.rec_gain
        return W

    def apply_stress_modulation(self, W, stress_level):
        """
        W shape: [post, pre]
        """

        if not torch.is_tensor(stress_level):
            stress_level = torch.tensor(stress_level, device=W.device)

        stress_level = stress_level.to(W.device).float()

        post_pl_pn = (self.is_pl & self.is_exc).view(-1, 1)
        post_il_pn = (self.is_il & self.is_exc).view(-1, 1)

        pre_exc = self.is_exc.view(1, -1)
        pre_inh = self.is_inh.view(1, -1)

        mask_pl_inh_to_pn = post_pl_pn & pre_inh
        mask_il_exc_to_pn = post_il_pn & pre_exc

        W_mod = W.clone()

        # PL: stronger inhibition onto PL pyramidal units
        W_mod = torch.where(
            mask_pl_inh_to_pn,
            W_mod * (1.0 + stress_level * F.softplus(self.pl_inh_gain_strength)),
            W_mod,
        )

        # IL: weaker excitation onto IL pyramidal units
        W_mod = torch.where(
            mask_il_exc_to_pn,
            W_mod * (1.0 - stress_level * torch.sigmoid(self.il_exc_supp_strength)),
            W_mod,
        )

        return W_mod

    def forward(self, u, stress_level=0.0, neuromod_signal=None, return_states=True):
        """
        u: [batch, time, input_dim]
        neuromod_signal: optional [batch, time, 1]
        """

        batch, T, _ = u.shape
        x = torch.zeros(batch, self.hidden_dim, device=u.device)

        W = self.dale_weight()
        W = self.apply_stress_modulation(W, stress_level)

        ys = []
        xs = []

        for t in range(T):
            drive = self.w_in(u[:, t]) + x @ W + self.bias

            if neuromod_signal is not None:
                gain = 1.0 + self.global_gain * neuromod_signal[:, t]
                drive = gain * drive

            x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(drive)
            y = self.readout(x)

            ys.append(y)

            if return_states:
                xs.append(x)

        ys = torch.stack(ys, dim=1)

        if return_states:
            xs = torch.stack(xs, dim=1)
            return ys, xs

        return ys



---

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuromodulatedDaleEIRNN(nn.Module):
    """
    EI-RNN with:
    - Dale constraint
    - IL / PL region split
    - pyramidal / PV-like inhibitory split
    - neuromodulation of excitation and inhibition
    - stress-specific PL and IL mechanisms from Rodrigues et al. 2024
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        p_exc=0.8,
        pl_fraction=0.5,
        dt=1.0,
        tau=20.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alpha = dt / tau

        n_pl = int(hidden_dim * pl_fraction)
        n_il = hidden_dim - n_pl

        self.register_buffer("is_pl", torch.zeros(hidden_dim).bool())
        self.register_buffer("is_il", torch.zeros(hidden_dim).bool())

        self.is_pl[:n_pl] = True
        self.is_il[n_pl:] = True

        n_exc = int(hidden_dim * p_exc)

        self.register_buffer("is_exc", torch.zeros(hidden_dim).bool())
        self.register_buffer("is_inh", torch.zeros(hidden_dim).bool())

        self.is_exc[:n_exc] = True
        self.is_inh[n_exc:] = True

        dale_sign = torch.ones(hidden_dim)
        dale_sign[self.is_inh] = -1.0
        self.register_buffer("dale_sign", dale_sign.view(1, hidden_dim))

        self.w_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_raw = nn.Parameter(0.05 * torch.randn(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.readout = nn.Linear(hidden_dim, output_dim)

        # Neuromodulation parameters.
        # stress_level = 0 means control.
        # stress_level = 1 means chronic-stress-like regime.
        self.pl_inh_gain_strength = nn.Parameter(torch.tensor(0.5))
        self.il_exc_supp_strength = nn.Parameter(torch.tensor(0.5))

        # Optional global neuromodulatory gain, e.g. arousal / NE / ACh.
        self.global_gain = nn.Parameter(torch.tensor(1.0))

    def dale_weight(self):
        w = F.softplus(self.w_raw)
        w = w * self.dale_sign
        w = w / self.hidden_dim**0.5
        return w

    def apply_stress_modulation(self, W, stress_level):
        """
        W shape: [post, pre]
        Columns are presynaptic neurons.
        Rows are postsynaptic neurons.

        PL mechanism:
            stress increases inhibition onto PL pyramidal neurons.

        IL mechanism:
            stress decreases excitation onto IL pyramidal neurons.

        PV-like inhibitory neurons are not directly modulated here.
        """

        if not torch.is_tensor(stress_level):
            stress_level = torch.tensor(stress_level, device=W.device)

        stress_level = stress_level.to(W.device)

        post_pl_pn = (self.is_pl & self.is_exc).view(-1, 1)
        post_il_pn = (self.is_il & self.is_exc).view(-1, 1)

        pre_exc = self.is_exc.view(1, -1)
        pre_inh = self.is_inh.view(1, -1)

        # Matrix masks
        pl_inh_to_pn = post_pl_pn & pre_inh
        il_exc_to_pn = post_il_pn & pre_exc

        W_mod = W.clone()

        # PL: stronger inhibitory input onto PL pyramidal neurons
        W_mod = torch.where(
            pl_inh_to_pn,
            W_mod * (1.0 + stress_level * F.softplus(self.pl_inh_gain_strength)),
            W_mod,
        )

        # IL: weaker excitatory input onto IL pyramidal neurons
        W_mod = torch.where(
            il_exc_to_pn,
            W_mod * (1.0 - stress_level * torch.sigmoid(self.il_exc_supp_strength)),
            W_mod,
        )

        return W_mod

    def forward(self, u, stress_level=0.0, neuromod_signal=None):
        """
        u: [batch, time, input_dim]

        stress_level:
            scalar or tensor in [0, 1]
            0 = control
            1 = chronic-stress-like modulation

        neuromod_signal:
            optional tensor [batch, time, 1]
            multiplicative gain signal
        """

        batch, time, _ = u.shape
        x = torch.zeros(batch, self.hidden_dim, device=u.device)

        W = self.dale_weight()
        W = self.apply_stress_modulation(W, stress_level)

        ys = []
        xs = []

        for t in range(time):
            drive = self.w_in(u[:, t]) + x @ W + self.bias

            if neuromod_signal is not None:
                gain = 1.0 + self.global_gain * neuromod_signal[:, t]
                drive = gain * drive

            x = (1.0 - self.alpha) * x + self.alpha * torch.tanh(drive)

            y = self.readout(x)

            ys.append(y)
            xs.append(x)

        return torch.stack(ys, dim=1), torch.stack(xs, dim=1)
