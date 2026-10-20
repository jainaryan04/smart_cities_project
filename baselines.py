"""
Reference baselines for the trajectory benchmark.
1. Constant-velocity (CV) and constant-acceleration (CA) physical extrapolation
2. Centralized model with zone-specific output heads (the fair centralized reference:
   pooled training that still knows which zone a window comes from)

The physical baselines are nn.Modules that take normalized inputs and return normalized
targets, so they run through the same evaluation code as the learned models.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import Seq2SeqLSTM
from metrics import evaluate_model_on_loader
from config import LEARNING_RATE, WEIGHT_DECAY, DEVICE, SAMPLE_DT_S


class KinematicExtrapolation(nn.Module):
    """
    Extrapolates the last observed relative positions.
    mode='cv': p(t) = v * t, with v from the last two observed positions.
    mode='ca': p(t) = v * t + 0.5 * a * t^2, with a from the last three positions.
    """
    def __init__(self, scalers, mode='cv', pred_steps=None, dt=SAMPLE_DT_S):
        super(KinematicExtrapolation, self).__init__()
        from config import PRED_STEPS
        self.mode = mode
        self.pred_steps = pred_steps or PRED_STEPS
        self.dt = dt
        for k in ['x_mean', 'x_std', 'y_mean', 'y_std']:
            self.register_buffer(k, torch.as_tensor(scalers[k], dtype=torch.float32))

    def forward(self, x, targets=None):
        pos = (x * self.x_std + self.x_mean)[..., :2]  # [B, T_obs, 2] in feet, relative to t=0
        v = (pos[:, -1] - pos[:, -2]) / self.dt
        t = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=x.dtype).view(1, -1, 1) * self.dt
        pred = pos[:, -1:].clone() + v.unsqueeze(1) * t
        if self.mode == 'ca' and pos.shape[1] >= 3:
            a = (pos[:, -1] - 2 * pos[:, -2] + pos[:, -3]) / self.dt ** 2
            # backward-difference velocity lags by dt/2; correct it with the acceleration
            pred = pred + (a * self.dt / 2).unsqueeze(1) * t + 0.5 * a.unsqueeze(1) * t ** 2
        return (pred - self.y_mean[:2]) / self.y_std[:2]


class ZoneHeadSeq2Seq(nn.Module):
    """Seq2SeqLSTM with one output head per zone; the base is shared by all zones."""
    def __init__(self, num_zones=3):
        super(ZoneHeadSeq2Seq, self).__init__()
        self.base = Seq2SeqLSTM()
        self.heads = nn.ModuleList([Seq2SeqLSTM().fc_out for _ in range(num_zones)])
        self.base.fc_out = nn.Identity()

    def forward(self, x, zone):
        return self.heads[zone](self.base(x))

    def for_zone(self, zone):
        """Returns a module with the standard forward(x) signature for evaluating one zone."""
        parent = self

        class _ZoneView(nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = parent

            def forward(self, x, targets=None):
                return self.parent(x, zone)

        return _ZoneView()


class CentralizedZoneHeadTrainer:
    """
    Trains ZoneHeadSeq2Seq on the pooled data of all zones. Each mini-batch comes from one
    zone (batches are interleaved across zones), so the matching head is updated.
    """
    def __init__(self, client_loaders, scalers, device=DEVICE, lr=LEARNING_RATE):
        self.device = device
        self.client_loaders = client_loaders
        self.scalers = scalers
        self.model = ZoneHeadSeq2Seq(num_zones=len(client_loaders)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.MSELoss()

    def train(self, epochs=12):
        print(f"\n--- Centralized + zone heads: {epochs} epochs ---", flush=True)
        history = {'epochs': [], 'train_losses': []}
        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            iters = {cid: iter(l['train']) for cid, l in self.client_loaders.items()}
            total, n = 0.0, 0
            while iters:
                for cid in list(iters.keys()):
                    try:
                        batch = next(iters[cid])
                    except StopIteration:
                        del iters[cid]
                        continue
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.model(inputs, cid), targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    total += loss.item()
                    n += 1
            history['epochs'].append(ep)
            history['train_losses'].append(total / max(1, n))
            print(f"[Zone-head Epoch {ep:02d}/{epochs:02d}] Loss: {total / max(1, n):.4f} | Time: {time.time() - t0:.2f}s", flush=True)
        return history, self.model


if __name__ == '__main__':
    import numpy as np
    from config import OBS_STEPS, PRED_STEPS
    scalers = {'x_mean': np.zeros(4), 'x_std': np.ones(4), 'y_mean': np.zeros(2), 'y_std': np.ones(2)}
    # A vehicle moving at 10 ft/s along y, sampled every SAMPLE_DT_S
    t_obs = (np.arange(OBS_STEPS) - (OBS_STEPS - 1)) * SAMPLE_DT_S
    x = np.zeros((1, OBS_STEPS, 4), dtype=np.float32)
    x[0, :, 1] = 10.0 * t_obs
    cv = KinematicExtrapolation(scalers, 'cv')(torch.tensor(x))
    expected = 10.0 * np.arange(1, PRED_STEPS + 1) * SAMPLE_DT_S
    assert np.allclose(cv[0, :, 1].numpy(), expected, atol=1e-4), cv
    # Constant acceleration 4 ft/s^2 from rest at t=-T: CA must be exact
    x[0, :, 1] = 0.5 * 4.0 * (t_obs - t_obs[0]) ** 2 - 0.5 * 4.0 * (0 - t_obs[0]) ** 2
    ca = KinematicExtrapolation(scalers, 'ca')(torch.tensor(x))
    tt = np.arange(1, PRED_STEPS + 1) * SAMPLE_DT_S
    exp_ca = 0.5 * 4.0 * (tt - t_obs[0]) ** 2 - 0.5 * 4.0 * (0 - t_obs[0]) ** 2
    assert np.allclose(ca[0, :, 1].numpy(), exp_ca, atol=1e-3), (ca, exp_ca)
    m = ZoneHeadSeq2Seq()
    out = m.for_zone(1)(torch.randn(4, OBS_STEPS, 4))
    assert out.shape == (4, PRED_STEPS, 2)
    print("Baseline self-tests passed.")
