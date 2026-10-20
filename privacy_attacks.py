"""
Privacy Attack & Defense Module for Federated Vehicle Trajectory Prediction.
Implements:
1. Deep Leakage from Gradients (DLG) — Trajectory Reconstruction Attack
2. Improved DLG (iDLG) — Enhanced with analytical label/target initialization
3. Membership Inference Attack (MIA)
4. Gradient clipping + Gaussian noise on observed gradients (one-shot perturbation;
   DP-SGD training with an accountant is in dp_training.py)
5. Privacy evaluation utilities
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from config import DEVICE, OBS_STEPS, PRED_STEPS, INPUT_DIM, OUTPUT_DIM
from models import Seq2SeqLSTM, get_model_parameters


class GradientReconstructionAttack:
    """
    Implements Deep Leakage from Gradients (DLG) adapted for continuous vehicle trajectory sequences.
    Given shared model gradients from a victim client, optimizes dummy trajectory inputs and targets
    to match the observed gradients.
    """
    def __init__(self, model, device=DEVICE):
        """
        model: the target model (Seq2SeqLSTM) with parameters at the state 
               BEFORE the victim client trained on it.
        """
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        self.device = device
        self.criterion = nn.MSELoss()

    def compute_target_gradient(self, real_input, real_target, shared_only=False):
        """
        Simulates the victim's gradient computation on real data.
        real_input: [1, T_obs, 4] (or [T_obs, 4]) — a single trajectory observation
        real_target: [1, T_pred, 2] (or [T_pred, 2]) — the ground truth future
        shared_only: bool, if True only compute/return gradients of shared parameters (excluding fc_out)
        Returns: list of gradient tensors
        """
        if real_input.dim() == 2:
            real_input = real_input.unsqueeze(0)
        if real_target.dim() == 2:
            real_target = real_target.unsqueeze(0)

        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        self.model.zero_grad()
        pred = self.model(real_input)
        loss = self.criterion(pred, real_target)
        if shared_only:
            target_params = [p for name, p in self.model.named_parameters() if 'fc_out' not in name]
        else:
            target_params = list(self.model.parameters())

        gradients = torch.autograd.grad(loss, target_params, create_graph=False)
        return [g.detach().clone() for g in gradients]

    def reconstruct(self, target_gradients, num_iterations=300, lr=0.1, 
                    tv_weight=1e-4, verbose=False, shared_only=False, batch_size=1):
        """
        Attempts to reconstruct the input trajectory and target from observed gradients.
        batch_size: number of windows behind the observed (batch-averaged) gradient.
        
        target_gradients: list of gradient tensors (the 'leaked' gradients)
        num_iterations: optimization iterations
        lr: learning rate for reconstruction
        tv_weight: total variation regularization weight for smooth trajectories
        shared_only: bool, whether attacker only observed shared parameters (FedPer scenario)
        
        Returns: (reconstructed_input, reconstructed_target, loss_history)
          reconstructed_input: [1, T_obs, INPUT_DIM]
          reconstructed_target: [1, T_pred, OUTPUT_DIM]
          loss_history: list of float loss values
        """
        target_gradients = [tg.to(self.device) for tg in target_gradients]

        # Initialize dummy data
        dummy_input = torch.randn(batch_size, OBS_STEPS, INPUT_DIM, device=self.device, requires_grad=True)
        dummy_target = torch.randn(batch_size, PRED_STEPS, OUTPUT_DIM, device=self.device, requires_grad=True)

        optimizer = optim.Adam([dummy_input, dummy_target], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

        loss_history = []

        for i in range(num_iterations):
            optimizer.zero_grad()
            self.model.zero_grad()

            # Forward pass with dummy data
            dummy_pred = self.model(dummy_input)
            dummy_loss = self.criterion(dummy_pred, dummy_target)
            if shared_only:
                target_params = [p for name, p in self.model.named_parameters() if 'fc_out' not in name]
            else:
                target_params = list(self.model.parameters())

            dummy_gradients = torch.autograd.grad(
                dummy_loss, target_params, create_graph=True
            )

            # Gradient matching loss (L2 norm distance)
            grad_loss = sum(
                ((dg - tg) ** 2).sum() for dg, tg in zip(dummy_gradients, target_gradients)
            )

            # Total variation regularization for trajectory smoothness
            tv_loss = torch.tensor(0.0, device=self.device)
            if tv_weight > 0 and dummy_input.shape[1] > 1:
                tv_loss = tv_weight * torch.sum(
                    (dummy_input[:, 1:, :2] - dummy_input[:, :-1, :2]) ** 2
                )

            total_loss = grad_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(grad_loss.item())

            if verbose and (i + 1) % 50 == 0:
                print(f"  [DLG Iter {i+1}/{num_iterations}] Grad Loss: {grad_loss.item():.6f}", flush=True)

        return (dummy_input.detach().cpu(), 
                dummy_target.detach().cpu(), 
                loss_history)

    def evaluate_reconstruction(self, original_input, original_target, 
                                reconstructed_input, reconstructed_target):
        """
        Computes reconstruction quality metrics.
        Returns dict with:
          - input_mse: MSE between original and reconstructed input trajectories
          - target_mse: MSE between original and reconstructed targets
          - input_cosine_sim: cosine similarity of flattened input vectors
          - target_cosine_sim: cosine similarity of flattened target vectors
          - position_error: average Euclidean distance of position coordinates (first 2 dims of
            input), in standardized units
          - endpoint_error: Euclidean distance of final predicted position (last step of target)
        """
        if original_input.dim() == 2:
            original_input = original_input.unsqueeze(0)
        if original_target.dim() == 2:
            original_target = original_target.unsqueeze(0)
        if reconstructed_input.dim() == 2:
            reconstructed_input = reconstructed_input.unsqueeze(0)
        if reconstructed_target.dim() == 2:
            reconstructed_target = reconstructed_target.unsqueeze(0)

        orig_in = original_input.cpu().float()
        orig_tg = original_target.cpu().float()
        rec_in = reconstructed_input.cpu().float()
        rec_tg = reconstructed_target.cpu().float()

        # With a batch, the order of reconstructed windows is arbitrary: match each
        # reconstruction to an original window by minimum position error.
        if orig_in.shape[0] > 1:
            from scipy.optimize import linear_sum_assignment
            cost = torch.cdist(orig_in[..., :2].reshape(orig_in.shape[0], -1),
                               rec_in[..., :2].reshape(rec_in.shape[0], -1)).numpy()
            _, col = linear_sum_assignment(cost)
            rec_in, rec_tg = rec_in[col], rec_tg[col]

        input_mse = float(torch.mean((orig_in - rec_in) ** 2).item())
        target_mse = float(torch.mean((orig_tg - rec_tg) ** 2).item())

        n = orig_in.shape[0]
        input_cos = float(nn.functional.cosine_similarity(orig_in.reshape(n, -1), rec_in.reshape(n, -1)).mean().item())
        target_cos = float(nn.functional.cosine_similarity(orig_tg.reshape(n, -1), rec_tg.reshape(n, -1)).mean().item())

        # Average Euclidean distance of (x, y) coordinates across all observation steps.
        # Inputs are standardized, so this is in standardized units, not metres.
        pos_dist = torch.norm(orig_in[..., :2] - rec_in[..., :2], dim=-1)  # [B, T_obs]
        position_error = float(pos_dist.mean().item())

        # Final predicted position endpoint Euclidean distance
        endpoint_dist = torch.norm(orig_tg[:, -1, :2] - rec_tg[:, -1, :2], dim=-1)  # [B]
        endpoint_error = float(endpoint_dist.mean().item())

        return {
            "input_mse": input_mse,
            "target_mse": target_mse,
            "input_cosine_sim": input_cos,
            "target_cosine_sim": target_cos,
            "position_error": position_error,
            "endpoint_error": endpoint_error
        }


class ImprovedGradientReconstructionAttack(GradientReconstructionAttack):
    """
    Improved DLG (iDLG) adapted for trajectory regression.
    Extracts prior target information from the output linear layer gradients
    to initialize dummy targets closer to the ground truth before DLG optimization.
    """
    def reconstruct(self, target_gradients, num_iterations=300, lr=0.1, 
                    tv_weight=1e-4, verbose=False, shared_only=False, batch_size=1):
        """
        Performs iDLG with output-gradient based target initialization.
        """
        target_gradients = [tg.to(self.device) for tg in target_gradients]

        # Analytical initialization for last linear bias gradient if available
        # In MSE loss: dL/db_out reflects (y_pred - y_real)
        # Using output gradient signs and magnitudes provides faster convergence
        init_target = torch.randn(batch_size, PRED_STEPS, OUTPUT_DIM, device=self.device)
        if not shared_only and len(target_gradients) > 0:
            last_bias_grad = target_gradients[-1]
            if last_bias_grad.numel() == OUTPUT_DIM:
                # Scale target initialization toward the negative bias gradient direction
                bias_dir = -last_bias_grad.view(1, 1, OUTPUT_DIM).repeat(batch_size, PRED_STEPS, 1)
                init_target = init_target + 0.1 * bias_dir

        dummy_input = torch.randn(batch_size, OBS_STEPS, INPUT_DIM, device=self.device, requires_grad=True)
        dummy_target = init_target.detach().clone().requires_grad_(True)

        optimizer = optim.Adam([dummy_input, dummy_target], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

        loss_history = []

        for i in range(num_iterations):
            optimizer.zero_grad()
            self.model.zero_grad()

            dummy_pred = self.model(dummy_input)
            dummy_loss = self.criterion(dummy_pred, dummy_target)
            if shared_only:
                target_params = [p for name, p in self.model.named_parameters() if 'fc_out' not in name]
            else:
                target_params = list(self.model.parameters())

            dummy_gradients = torch.autograd.grad(
                dummy_loss, target_params, create_graph=True
            )

            grad_loss = sum(
                ((dg - tg) ** 2).sum() for dg, tg in zip(dummy_gradients, target_gradients)
            )

            tv_loss = torch.tensor(0.0, device=self.device)
            if tv_weight > 0 and dummy_input.shape[1] > 1:
                tv_loss = tv_weight * torch.sum(
                    (dummy_input[:, 1:, :2] - dummy_input[:, :-1, :2]) ** 2
                )

            total_loss = grad_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(grad_loss.item())

            if verbose and (i + 1) % 50 == 0:
                print(f"  [iDLG Iter {i+1}/{num_iterations}] Grad Loss: {grad_loss.item():.6f}", flush=True)

        return (dummy_input.detach().cpu(), 
                dummy_target.detach().cpu(), 
                loss_history)


class MembershipInferenceAttack:
    """
    Determines whether a specific vehicle trajectory was part of a model's training set.
    Uses the loss-threshold approach: member trajectories typically incur lower prediction loss.
    """
    def __init__(self, target_model, device=DEVICE):
        self.model = copy.deepcopy(target_model).to(device)
        self.model.eval()
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')

    def compute_sample_loss(self, input_tensor, target_tensor):
        """
        Compute per-sample scalar MSE loss for a single trajectory.
        input_tensor: [1, T_obs, INPUT_DIM] or [T_obs, INPUT_DIM]
        target_tensor: [1, T_pred, OUTPUT_DIM] or [T_pred, OUTPUT_DIM]
        Returns: float scalar loss value
        """
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)
            loss = self.criterion(pred, target_tensor)
            # Average across sequence length and coordinate dimensions for each sample
            sample_loss = loss.mean(dim=(1, 2)).item()
        return sample_loss

    def evaluate(self, member_loader, non_member_loader, num_samples=200):
        """
        Evaluates MIA success using training (member) and holdout (non-member) data.
        
        member_loader: DataLoader with trajectories that WERE in training
        non_member_loader: DataLoader with trajectories NOT in training (test set or other zone)
        num_samples: maximum number of samples to evaluate from each loader
        
        Returns dict with:
          - accuracy
          - precision
          - recall
          - f1
          - auc
          - optimal_threshold
          - member_mean_loss
          - non_member_mean_loss
        """
        member_losses = []
        non_member_losses = []

        # Collect member sample losses
        for batch in member_loader:
            inputs, targets = batch[0], batch[1]
            for i in range(inputs.size(0)):
                loss = self.compute_sample_loss(inputs[i], targets[i])
                member_losses.append(loss)
                if len(member_losses) >= num_samples:
                    break
            if len(member_losses) >= num_samples:
                break

        # Collect non-member sample losses
        for batch in non_member_loader:
            inputs, targets = batch[0], batch[1]
            for i in range(inputs.size(0)):
                loss = self.compute_sample_loss(inputs[i], targets[i])
                non_member_losses.append(loss)
                if len(non_member_losses) >= num_samples:
                    break
            if len(non_member_losses) >= num_samples:
                break

        n_members = len(member_losses)
        n_non_members = len(non_member_losses)

        if n_members == 0 or n_non_members == 0:
            return {
                "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5,
                "optimal_threshold": 0.0, "member_mean_loss": 0.0, "non_member_mean_loss": 0.0
            }

        y_true = np.array([1] * n_members + [0] * n_non_members)
        losses = np.array(member_losses + non_member_losses)

        # Members exhibit lower loss -> higher score = -loss for ROC curve calculation
        member_scores = -losses
        try:
            auc = float(roc_auc_score(y_true, member_scores))
        except Exception:
            auc = 0.5

        # Find optimal decision threshold on loss minimizing total classification error
        # Predict Member (1) if loss <= threshold, else Non-Member (0)
        sorted_thresholds = np.unique(np.sort(losses))
        best_acc = -1.0
        best_thresh = sorted_thresholds[0]

        for thresh in sorted_thresholds:
            y_pred = (losses <= thresh).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        optimal_pred = (losses <= best_thresh).astype(int)
        accuracy = float(accuracy_score(y_true, optimal_pred))
        precision = float(precision_score(y_true, optimal_pred, zero_division=0))
        recall = float(recall_score(y_true, optimal_pred, zero_division=0))
        f1 = float(f1_score(y_true, optimal_pred, zero_division=0))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "optimal_threshold": float(best_thresh),
            "member_mean_loss": float(np.mean(member_losses)),
            "non_member_mean_loss": float(np.mean(non_member_losses))
        }


class DifferentialPrivacyDefense:
    """
    Applies differential privacy to model updates via gradient clipping and noise addition.
    Implements the DP-SGD mechanism adapted for federated learning.
    """
    def __init__(self, clip_norm=1.0, noise_multiplier=1.0):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def clip_and_noise_gradients(self, model):
        """
        Clips per-parameter gradient norm and adds calibrated Gaussian noise in-place.
        Returns total_norm before clipping.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        if self.noise_multiplier > 0:
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        0.0,
                        self.clip_norm * self.noise_multiplier,
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad += noise
        return total_norm

    def add_noise_to_weights(self, state_dict, sensitivity=None):
        """
        Adds Gaussian noise directly to model weight updates (for federated setting).
        Used when the server or client wants to perturb the uploaded weights.
        Returns noised state dict.
        """
        noised = {}
        for key, tensor in state_dict.items():
            if tensor.is_floating_point() and self.noise_multiplier > 0:
                noise_std = self.clip_norm * self.noise_multiplier
                noise = torch.normal(0.0, noise_std, size=tensor.shape, device=tensor.device)
                noised[key] = tensor + noise
            else:
                noised[key] = tensor.clone()
        return noised

    # Alias for convenient terminology
    add_noise_to_state_dict = add_noise_to_weights

    @staticmethod
    def compute_epsilon(noise_multiplier, num_steps, delta=1e-5, sample_rate=0.01):
        """
        Privacy budget of num_steps DP-SGD steps (Poisson sampling rate sample_rate,
        per-example clipping, noise multiplier sigma), from the RDP accountant in
        dp_training.py. Only meaningful for models actually trained with DP-SGD.
        """
        from dp_training import compute_epsilon
        return compute_epsilon(sample_rate, noise_multiplier, num_steps, delta)


def run_privacy_evaluation(model_class, client_loaders, scalers, 
                           fl_method='fedavg', num_attack_samples=10,
                           dp_noise_multipliers=[0.0, 0.5, 1.0, 2.0],
                           device=DEVICE, model=None, attack_batch_size=1,
                           clip_norm=1.0, mia_samples=200, victim_client=0):
    """
    Privacy evaluation pipeline.

    model: the model under attack (normally the final trained model of the FL method).
           If None, a freshly initialized model_class() is attacked, as in the earlier
           experiments; results on an untrained model say little about deployed models.
    1. DLG on num_attack_samples batches of attack_batch_size training windows of the
       victim client. With noise, the observed batch gradient is clipped to clip_norm and
       N(0, sigma^2 clip_norm^2) is added: a one-shot perturbation, not DP-SGD training,
       so no epsilon is reported here (see dp_training.py for DP training + accountant).
    2. Loss-threshold MIA: victim client training windows (members) vs test windows.
    Reconstruction errors are in standardized input units.
    """
    shared_only = (fl_method.lower() == 'fedper')
    method_title = "FedPer - Shared Gradients Only" if shared_only else f"{fl_method.upper()} - Full Gradients"
    print(f"\n=======================================================", flush=True)
    print(f"  Running Privacy Attack & Defense Evaluation ({method_title})", flush=True)
    print(f"  Model: {'trained' if model is not None else 'freshly initialized'} | "
          f"attack batch size: {attack_batch_size} | batches: {num_attack_samples}", flush=True)
    print(f"=======================================================", flush=True)

    if model is None:
        model = model_class().to(device)
    model = copy.deepcopy(model).to(device)
    model.eval()

    victim_data = client_loaders[victim_client]
    train_loader = victim_data['train']
    test_loader = victim_data['test']

    victim_batches = []
    pool_in, pool_tg = [], []
    for batch in train_loader:
        pool_in.append(batch[0])
        pool_tg.append(batch[1])
        if sum(len(x) for x in pool_in) >= num_attack_samples * attack_batch_size:
            break
    pool_in, pool_tg = torch.cat(pool_in), torch.cat(pool_tg)
    for k in range(min(num_attack_samples, len(pool_in) // attack_batch_size)):
        sl = slice(k * attack_batch_size, (k + 1) * attack_batch_size)
        victim_batches.append((pool_in[sl].clone(), pool_tg[sl].clone()))

    attacker = GradientReconstructionAttack(model, device=device)

    reconstruction_results_by_dp = {}

    for nm in dp_noise_multipliers:
        print(f"\n[Privacy Eval] Reconstruction ({fl_method}) with gradient noise multiplier: {nm}...", flush=True)
        sample_metrics = []

        for real_in, real_tg in victim_batches:
            clean_grads = attacker.compute_target_gradient(real_in, real_tg, shared_only=shared_only)

            defended_grads = clean_grads
            if nm > 0:
                total_norm = torch.sqrt(sum((g ** 2).sum() for g in clean_grads))
                factor = min(1.0, clip_norm / (float(total_norm) + 1e-12))
                defended_grads = [g * factor + torch.normal(0.0, clip_norm * nm, size=g.shape, device=g.device)
                                  for g in clean_grads]

            rec_in, rec_tg, _ = attacker.reconstruct(
                defended_grads, num_iterations=150, lr=0.1, tv_weight=1e-4, verbose=False,
                shared_only=shared_only, batch_size=real_in.shape[0]
            )

            metrics = attacker.evaluate_reconstruction(real_in, real_tg, rec_in, rec_tg)
            sample_metrics.append(metrics)

        avg_metrics = {
            k: float(np.mean([m[k] for m in sample_metrics]))
            for k in sample_metrics[0].keys()
        }
        reconstruction_results_by_dp[f"dp_sigma_{nm}"] = avg_metrics

        print(f"  -> Pos Error (std. units): {avg_metrics['position_error']:.4f} | "
              f"Input Cosine: {avg_metrics['input_cosine_sim']:.4f}", flush=True)

    print(f"\n[Privacy Eval] Evaluating Membership Inference Attack (MIA) for {fl_method}...", flush=True)
    mia = MembershipInferenceAttack(model, device=device)
    mia_results = mia.evaluate(train_loader, test_loader, num_samples=mia_samples)

    print(f"  -> MIA AUC: {mia_results['auc']:.4f} | "
          f"Accuracy: {mia_results['accuracy']:.4f} | "
          f"Member Mean Loss: {mia_results['member_mean_loss']:.6f} | "
          f"Non-member Mean Loss: {mia_results['non_member_mean_loss']:.6f}", flush=True)

    results = {
        "fl_method": fl_method,
        "shared_only": shared_only,
        "num_attack_batches": len(victim_batches),
        "attack_batch_size": attack_batch_size,
        "reconstruction_defense": reconstruction_results_by_dp,
        "membership_inference": mia_results
    }
    return results


def run_membership_inference(model, client_loaders, device=DEVICE, num_samples=200):
    """Loss-threshold MIA on every client: its training windows vs its test windows."""
    out = {}
    for cid, loaders in client_loaders.items():
        out[cid] = MembershipInferenceAttack(model, device=device).evaluate(
            loaders['train'], loaders['test'], num_samples=num_samples)
    return out


if __name__ == '__main__':
    print("Testing Privacy Attacks & Defenses module...")
    torch.manual_seed(42)

    # Quick smoke test with dummy model and synthetic trajectory data
    model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=64, num_layers=2)
    device = torch.device("cpu")
    model.to(device)

    dummy_input = torch.randn(1, OBS_STEPS, INPUT_DIM)
    dummy_target = torch.randn(1, PRED_STEPS, OUTPUT_DIM)

    # 1. GradientReconstructionAttack test
    print("\n1. Testing GradientReconstructionAttack (DLG)...")
    attacker = GradientReconstructionAttack(model, device=device)
    target_grads = attacker.compute_target_gradient(dummy_input, dummy_target)
    print(f"   Target gradients computed for {len(target_grads)} parameters.")
    
    rec_in, rec_tg, losses = attacker.reconstruct(target_grads, num_iterations=20, lr=0.1, verbose=True)
    metrics = attacker.evaluate_reconstruction(dummy_input, dummy_target, rec_in, rec_tg)
    print(f"   Reconstruction Metrics: Input MSE: {metrics['input_mse']:.4f}, Pos Error: {metrics['position_error']:.4f}")

    # 2. ImprovedGradientReconstructionAttack test
    print("\n2. Testing ImprovedGradientReconstructionAttack (iDLG)...")
    i_attacker = ImprovedGradientReconstructionAttack(model, device=device)
    i_rec_in, i_rec_tg, _ = i_attacker.reconstruct(target_grads, num_iterations=20, lr=0.1, verbose=False)
    i_metrics = i_attacker.evaluate_reconstruction(dummy_input, dummy_target, i_rec_in, i_rec_tg)
    print(f"   iDLG Metrics: Input MSE: {i_metrics['input_mse']:.4f}, Cosine Sim: {i_metrics['input_cosine_sim']:.4f}")

    # 3. Differential Privacy Defense test
    print("\n3. Testing DifferentialPrivacyDefense...")
    dp = DifferentialPrivacyDefense(clip_norm=1.0, noise_multiplier=0.5)
    eps = dp.compute_epsilon(noise_multiplier=0.5, num_steps=50)
    print(f"   Differential Privacy Epsilon (sigma=0.5, 50 steps): {eps:.4f}")

    # 4. Membership Inference Attack test
    print("\n4. Testing MembershipInferenceAttack...")
    mia = MembershipInferenceAttack(model, device=device)
    sample_loss = mia.compute_sample_loss(dummy_input, dummy_target)
    print(f"   Single trajectory loss: {sample_loss:.6f}")

    from torch.utils.data import TensorDataset, DataLoader
    member_ds = TensorDataset(torch.randn(10, OBS_STEPS, INPUT_DIM), torch.randn(10, PRED_STEPS, OUTPUT_DIM))
    non_member_ds = TensorDataset(torch.randn(10, OBS_STEPS, INPUT_DIM), torch.randn(10, PRED_STEPS, OUTPUT_DIM))
    m_loader = DataLoader(member_ds, batch_size=2)
    nm_loader = DataLoader(non_member_ds, batch_size=2)
    mia_res = mia.evaluate(m_loader, nm_loader, num_samples=10)
    print(f"   MIA test: AUC={mia_res['auc']:.4f}, Accuracy={mia_res['accuracy']:.4f}")

    print("\n[SUCCESS] All Privacy Attack & Defense components verified!")
