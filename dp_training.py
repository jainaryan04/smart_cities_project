"""
Differentially private federated training (DP-SGD on each client) with a Renyi DP accountant.

- DPFederatedClient: Poisson-subsampled mini-batches, per-example gradient clipping to
  norm C, Gaussian noise N(0, sigma^2 C^2) added to the summed gradient (Abadi et al., 2016).
- compute_rdp / get_privacy_spent: RDP of the subsampled Gaussian mechanism at integer
  orders (Mironov, Talwar and Zhang, 2019) and conversion to (epsilon, delta)
  (Balle et al., 2020), matching opacus.accountants.analysis.rdp.

The guarantee is per training window (record-level DP), not per vehicle: one vehicle
contributes several overlapping windows, so vehicle-level privacy is weaker.
"""

import math
import numpy as np
import torch
import torch.optim as optim
from fl_framework import FederatedClient
from config import LEARNING_RATE, WEIGHT_DECAY, DEVICE, BATCH_SIZE

DEFAULT_ORDERS = list(range(2, 257))


def _log_add(a, b):
    lo, hi = min(a, b), max(a, b)
    if lo == -np.inf:
        return hi
    return math.log1p(math.exp(lo - hi)) + hi


def _compute_log_a_int(q, sigma, alpha):
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef = (math.log(math.comb(alpha, i)) + i * math.log(q) + (alpha - i) * math.log(1 - q))
        log_a = _log_add(log_a, log_coef + (i * i - i) / (2 * sigma ** 2))
    return log_a


def compute_rdp(q, sigma, steps, orders=DEFAULT_ORDERS):
    """RDP of `steps` compositions of the Poisson-subsampled Gaussian mechanism."""
    rdp = []
    for alpha in orders:
        if sigma == 0:
            rdp.append(np.inf)
        elif q == 0:
            rdp.append(0.0)
        elif q == 1.0:
            rdp.append(alpha / (2 * sigma ** 2))
        else:
            rdp.append(_compute_log_a_int(q, sigma, int(alpha)) / (alpha - 1))
    return np.array(rdp) * steps


def get_privacy_spent(orders, rdp, delta):
    """Converts RDP to (epsilon, best order) for the given delta."""
    orders = np.atleast_1d(np.array(orders, dtype=float))
    rdp = np.atleast_1d(rdp)
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    if np.isnan(eps).all():
        return np.inf, np.nan
    idx = int(np.nanargmin(eps))
    return float(eps[idx]), float(orders[idx])


def compute_epsilon(sample_rate, noise_multiplier, steps, delta=1e-5, orders=DEFAULT_ORDERS):
    if noise_multiplier <= 0:
        return float('inf')
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    return get_privacy_spent(orders, rdp, delta)[0]


class DPFederatedClient(FederatedClient):
    """
    FedAvg client that trains with DP-SGD. Each local step draws a Poisson sample of the
    client's training windows with rate q = batch_size / N_k, clips every per-example
    gradient to clip_norm, sums, adds Gaussian noise and divides by the expected batch size.
    """
    def __init__(self, client_id, name, train_loader, val_loader, test_loader,
                 model_class, scalers, device=DEVICE, lr=LEARNING_RATE,
                 noise_multiplier=1.0, clip_norm=1.0, batch_size=BATCH_SIZE, seed=0):
        super().__init__(client_id, name, train_loader, val_loader, test_loader,
                         model_class, scalers, device=device, lr=lr)
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.expected_batch = batch_size
        self.sample_rate = min(1.0, batch_size / self.sample_count)
        self.steps = 0
        self.generator = torch.Generator().manual_seed(seed + client_id)
        ds = train_loader.dataset
        self.inputs, self.targets = ds.inputs, ds.targets

    def train_epoch(self, fedprox_mu=0.0, global_params=None):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        params = [p for p in self.model.parameters() if p.requires_grad]
        n = self.sample_count
        steps_per_epoch = max(1, int(round(1.0 / self.sample_rate)))
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            mask = torch.rand(n, generator=self.generator) < self.sample_rate
            idx = torch.nonzero(mask).flatten()
            summed = [torch.zeros_like(p) for p in params]
            for i in idx.tolist():
                x = self.inputs[i:i + 1].to(self.device)
                y = self.targets[i:i + 1].to(self.device)
                loss = self.criterion(self.model(x), y)
                grads = torch.autograd.grad(loss, params)
                norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
                factor = min(1.0, self.clip_norm / (float(norm) + 1e-12))
                for s, g in zip(summed, grads):
                    s.add_(g * factor)
                total_loss += loss.item()
            optimizer.zero_grad()
            for p, s in zip(params, summed):
                noise = torch.normal(0.0, self.noise_multiplier * self.clip_norm, size=s.shape,
                                     generator=self.generator).to(self.device)
                p.grad = (s + noise) / self.expected_batch
            optimizer.step()
            self.steps += 1

        return total_loss / max(1, n)

    def epsilon(self, delta=1e-5):
        return compute_epsilon(self.sample_rate, self.noise_multiplier, self.steps, delta)


if __name__ == '__main__':
    eps = compute_epsilon(sample_rate=128 / 1430, noise_multiplier=1.0, steps=144, delta=1e-5)
    print(f"epsilon(q=128/1430, sigma=1.0, T=144, delta=1e-5) = {eps:.3f}")
    try:
        from opacus.accountants.analysis.rdp import compute_rdp as o_rdp, get_privacy_spent as o_spent
        o_eps = o_spent(orders=DEFAULT_ORDERS, rdp=o_rdp(q=128 / 1430, noise_multiplier=1.0, steps=144,
                                                          orders=DEFAULT_ORDERS), delta=1e-5)[0]
        print(f"opacus reference (same orders) = {o_eps:.3f}")
        assert abs(o_eps - eps) < 1e-6
    except ImportError:
        print("opacus not installed; reference check skipped")
    assert compute_epsilon(0.1, 0.0, 10) == float('inf')
    print("DP accountant self-test passed.")
