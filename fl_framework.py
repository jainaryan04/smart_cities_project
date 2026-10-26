"""
Federated Learning Simulation Framework for Smart City Trajectory Prediction.
Algorithms:
1. Federated Averaging (FedAvg - McMahan et al.)
2. Federated Proximal (FedProx - Li et al.) for Non-IID resilience
3. Centralized Pooled Training (Upper Bound Benchmark)
4. Isolated Local Training (Data Silo Baseline)
5. Communication Overhead & Privacy Tracking
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import get_model_parameters, set_model_parameters, clone_model, compute_model_size_kb
from metrics import evaluate_model_on_loader
from config import LEARNING_RATE, WEIGHT_DECAY, DEVICE, FEDPROX_MU


class FederatedClient:
    """
    Edge Client representing a Roadside Unit (RSU) or Traffic Monitoring Station.
    Holds private local trajectory data, performs on-device training, and communicates weights.
    """
    def __init__(self, client_id, name, train_loader, val_loader, test_loader,
                 model_class, scalers, device=DEVICE, lr=LEARNING_RATE):
        self.client_id = client_id
        self.name = name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scalers = scalers
        self.device = device
        self.lr = lr
        
        self.model = model_class().to(device)
        self.criterion = nn.MSELoss()
        self.sample_count = len(train_loader.dataset)

    def get_parameters(self):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters_dict):
        set_model_parameters(self.model, parameters_dict)

    def train_epoch(self, fedprox_mu=0.0, global_params=None):
        """Trains the local model for 1 epoch with optional FedProx regularization."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        total_loss = 0.0
        num_batches = 0

        global_param_tensors = {}
        if fedprox_mu > 0.0 and global_params is not None:
            for k, v in global_params.items():
                global_param_tensors[k] = v.to(self.device)

        for batch in self.train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            
            predictions = self.model(inputs, targets)
            loss = self.criterion(predictions, targets)

            if fedprox_mu > 0.0 and global_param_tensors:
                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_param_tensors:
                        prox_term += torch.sum((param - global_param_tensors[name]) ** 2)
                loss = loss + (fedprox_mu / 2.0) * prox_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def local_train(self, epochs=2, fedprox_mu=0.0, global_params=None):
        """Runs multiple local training epochs."""
        losses = []
        for ep in range(epochs):
            loss = self.train_epoch(fedprox_mu=fedprox_mu, global_params=global_params)
            losses.append(loss)
        return float(np.mean(losses))

    def evaluate(self, split='test'):
        """Evaluates local model on chosen split using ADE/FDE/RMSE metrics."""
        loader = self.test_loader if split == 'test' else self.val_loader
        return evaluate_model_on_loader(self.model, loader, self.scalers, self.device)


class FederatedServer:
    """
    Federated Learning Orchestrator.
    Manages global rounds, weight aggregation (FedAvg/FedProx), and global evaluations.
    """
    def __init__(self, model_class, clients, centralized_test_loader, scalers, device=DEVICE):
        self.device = device
        self.global_model = model_class().to(device)
        self.clients = clients
        self.test_loader = centralized_test_loader
        self.scalers = scalers
        self.model_size_kb = compute_model_size_kb(self.global_model)

    def get_parameters(self):
        return get_model_parameters(self.global_model)

    def set_parameters(self, parameters_dict):
        set_model_parameters(self.global_model, parameters_dict)

    def aggregate_fedavg(self, client_weights_list, sample_counts):
        """
        Federated Averaging Aggregation:
        w_global = sum_{k=1}^K (n_k / N_total) * w_k
        """
        total_samples = sum(sample_counts)
        client_shares = [n / total_samples for n in sample_counts]
        
        aggregated_dict = {}
        for key in client_weights_list[0].keys():
            weighted_tensor = torch.zeros_like(client_weights_list[0][key], dtype=torch.float32)
            for i, client_params in enumerate(client_weights_list):
                weighted_tensor += client_shares[i] * client_params[key].to(torch.float32)
            
            if client_weights_list[0][key].dtype != torch.float32:
                weighted_tensor = weighted_tensor.to(client_weights_list[0][key].dtype)
            aggregated_dict[key] = weighted_tensor

        self.set_parameters(aggregated_dict)
        return aggregated_dict

    def run_federated_training(self, num_rounds=12, local_epochs=2, fedprox_mu=0.0):
        """
        Executes the full Federated Learning simulation.
        Returns detailed history of convergence, metrics, and communication overhead.
        """
        print(f"\n=======================================================", flush=True)
        print(f"Starting Federated Learning Training ({'FedAvg' if fedprox_mu==0 else 'FedProx'})", flush=True)
        print(f"Rounds: {num_rounds}, Local Epochs/Round: {local_epochs}, Clients: {len(self.clients)}", flush=True)
        print(f"Model Size: {self.model_size_kb:.2f} KB | Total Parameters: {sum(p.numel() for p in self.global_model.parameters()):,}", flush=True)
        print(f"=======================================================\n", flush=True)

        history = {
            'rounds': [],
            'train_losses': [],
            'global_ade_m': [],
            'global_fde_m': [],
            'global_rmse_m': [],
            'client_ade_m': {client.client_id: [] for client in self.clients},
            'client_fde_m': {client.client_id: [] for client in self.clients},
            'cum_comm_mb': [],
            'round_times': []
        }

        total_comm_bytes = 0

        for r in range(1, num_rounds + 1):
            t_start = time.time()
            global_params = self.get_parameters()

            client_weights = []
            client_samples = []
            round_losses = []

            # Client Local Training
            for client in self.clients:
                client.set_parameters(copy.deepcopy(global_params))
                total_comm_bytes += self.model_size_kb * 1024

                local_loss = client.local_train(epochs=local_epochs, fedprox_mu=fedprox_mu, global_params=global_params)
                round_losses.append(local_loss)

                updated_params = client.get_parameters()
                client_weights.append(updated_params)
                client_samples.append(client.sample_count)
                total_comm_bytes += self.model_size_kb * 1024

            # Server Aggregation (FedAvg)
            self.aggregate_fedavg(client_weights, client_samples)
            
            # Global Evaluation on Centralized Test Set
            global_metrics = evaluate_model_on_loader(self.global_model, self.test_loader, self.scalers, self.device)
            
            # Per-Client Test Set Evaluation
            client_evals = {}
            for client in self.clients:
                c_metrics = evaluate_model_on_loader(self.global_model, client.test_loader, self.scalers, self.device)
                client_evals[client.client_id] = c_metrics
                history['client_ade_m'][client.client_id].append(c_metrics['ADE_m'])
                history['client_fde_m'][client.client_id].append(c_metrics['FDE_m'])

            elapsed = time.time() - t_start
            cum_mb = total_comm_bytes / (1024 * 1024)

            history['rounds'].append(r)
            history['train_losses'].append(float(np.mean(round_losses)))
            history['global_ade_m'].append(global_metrics['ADE_m'])
            history['global_fde_m'].append(global_metrics['FDE_m'])
            history['global_rmse_m'].append(global_metrics['RMSE_m'])
            history['cum_comm_mb'].append(cum_mb)
            history['round_times'].append(elapsed)

            client_ade_str = " | ".join([f"C{client.client_id}({client.name}): {client_evals[client.client_id]['ADE_m']:.3f}m" for client in self.clients])
            print(f"[Round {r:02d}/{num_rounds:02d}] "
                  f"Avg Loss: {np.mean(round_losses):.4f} | "
                  f"Global ADE: {global_metrics['ADE_m']:.3f}m ({global_metrics['ADE_ft']:.2f}ft) | "
                  f"Global FDE: {global_metrics['FDE_m']:.3f}m | "
                  f"[{client_ade_str}] | Time: {elapsed:.2f}s", flush=True)

        print(f"\n[FL Complete] Total Cumulative Comm: {cum_mb:.2f} MB | Final Global ADE: {history['global_ade_m'][-1]:.3f}m | Final FDE: {history['global_fde_m'][-1]:.3f}m", flush=True)
        return history


class CentralizedTrainer:
    """Trains a single model on pooled centralized data across all locations (Upper Bound)."""
    def __init__(self, model_class, cent_loaders, scalers, device=DEVICE, lr=LEARNING_RATE):
        self.device = device
        self.model = model_class().to(device)
        self.train_loader = cent_loaders['train']
        self.val_loader = cent_loaders['val']
        self.test_loader = cent_loaders['test']
        self.scalers = scalers
        self.lr = lr
        self.criterion = nn.MSELoss()

    def train(self, epochs=10):
        print(f"\n=======================================================", flush=True)
        print(f"Starting Centralized Training (Pooled Data Upper Bound)", flush=True)
        print(f"Epochs: {epochs}, Total Train Samples: {len(self.train_loader.dataset):,}", flush=True)
        print(f"=======================================================\n", flush=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        history = {'epochs': [], 'train_losses': [], 'val_ade_m': [], 'test_ade_m': [], 'test_fde_m': []}

        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for inputs, targets, _ in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                preds = self.model(inputs, targets)
                loss = self.criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            val_metrics = evaluate_model_on_loader(self.model, self.val_loader, self.scalers, self.device)
            test_metrics = evaluate_model_on_loader(self.model, self.test_loader, self.scalers, self.device)

            history['epochs'].append(ep)
            history['train_losses'].append(avg_loss)
            history['val_ade_m'].append(val_metrics['ADE_m'])
            history['test_ade_m'].append(test_metrics['ADE_m'])
            history['test_fde_m'].append(test_metrics['FDE_m'])

            print(f"[Centralized Epoch {ep:02d}/{epochs:02d}] "
                  f"Loss: {avg_loss:.4f} | Val ADE: {val_metrics['ADE_m']:.3f}m | "
                  f"Test ADE: {test_metrics['ADE_m']:.3f}m ({test_metrics['ADE_ft']:.2f}ft) | "
                  f"Test FDE: {test_metrics['FDE_m']:.3f}m | Time: {time.time()-t0:.2f}s", flush=True)

        return history, self.model


class IsolatedTrainer:
    """Trains independent models on each client in isolation without weight sharing (Data Silo Baseline)."""
    def __init__(self, model_class, client_loaders, scalers, device=DEVICE, lr=LEARNING_RATE):
        self.device = device
        self.model_class = model_class
        self.client_loaders = client_loaders
        self.scalers = scalers
        self.lr = lr
        self.models = {}

    def train_all(self, epochs=10):
        print(f"\n=======================================================", flush=True)
        print(f"Starting Isolated Local Training (Data Silo Baselines)", flush=True)
        print(f"Training {len(self.client_loaders)} independent models for {epochs} epochs each", flush=True)
        print(f"=======================================================\n", flush=True)

        results = {}
        for cid, loaders in self.client_loaders.items():
            print(f"--- Training Isolated Model for Client {cid}: {loaders['name']} ---", flush=True)
            model = self.model_class().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
            criterion = nn.MSELoss()

            for ep in range(1, epochs + 1):
                model.train()
                total_loss, num_b = 0.0, 0
                for inputs, targets, _ in loaders['train']:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    preds = model(inputs, targets)
                    loss = criterion(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    total_loss += loss.item()
                    num_b += 1

            self.models[cid] = model
            own_metrics = evaluate_model_on_loader(model, loaders['test'], self.scalers, self.device)
            results[cid] = own_metrics
            print(f"Isolated Client {cid} ({loaders['name']}) -> Own Test ADE: {own_metrics['ADE_m']:.3f}m, FDE: {own_metrics['FDE_m']:.3f}m", flush=True)

        return self.models, results


class CentralizedZoneAwareTrainer:
    """
    Trains a Centralized Zone-Aware model with shared backbone and zone-specific heads.
    Provides a fair comparison against FedPer under matched training budget.
    """
    def __init__(self, model_class, client_loaders, scalers, device=DEVICE, lr=LEARNING_RATE):
        self.device = device
        self.client_loaders = client_loaders
        self.scalers = scalers
        self.lr = lr
        self.model = model_class().to(device)
        self.criterion = nn.MSELoss()

    def train(self, epochs=12):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        num_clients = len(self.client_loaders)
        for ep in range(1, epochs + 1):
            self.model.train()
            iterators = [iter(self.client_loaders[cid]['train']) for cid in range(num_clients)]
            active = list(range(num_clients))
            while active:
                for cid in list(active):
                    try:
                        batch = next(iterators[cid])
                        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                        optimizer.zero_grad()
                        preds = self.model(inputs, zone_id=cid)
                        loss = self.criterion(preds, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        optimizer.step()
                    except StopIteration:
                        active.remove(cid)
        return self.model


# =============================================================
# Personalized Federated Learning: FedPer & pFedMe
# =============================================================

class FedPerClient(FederatedClient):
    """
    FedPer Client (Federated Personalization - Arivazhagan et al.).
    Maintains a shared base representation (encoder + decoder_lstm) and
    a private personal regression head (fc_out).
    """
    def __init__(self, client_id, name, train_loader, val_loader, test_loader,
                 model_class, scalers, device=DEVICE, lr=LEARNING_RATE):
        super().__init__(client_id, name, train_loader, val_loader, test_loader,
                         model_class, scalers, device, lr)

    def get_shared_parameters(self):
        """Returns only encoder and decoder_lstm parameters (keys NOT containing 'fc_out')."""
        return {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items() if 'fc_out' not in k}

    def get_personal_parameters(self):
        """Returns only fc_out parameters (keys containing 'fc_out')."""
        return {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items() if 'fc_out' in k}

    def set_shared_parameters(self, shared_params):
        """Loads ONLY the shared parameters into the model, keeping personal head untouched."""
        current_state = self.model.state_dict()
        current_state.update(shared_params)  # only updates shared keys
        self.model.load_state_dict(current_state)


class FedPerServer(FederatedServer):
    """
    FedPer Server Orchestrator.
    Coordinates federated learning across clients by communicating and aggregating
    only the shared temporal representation layers while preserving private client heads.
    """
    def __init__(self, model_class, clients, centralized_test_loader, scalers, device=DEVICE):
        super().__init__(model_class, clients, centralized_test_loader, scalers, device)
        shared_bytes = sum(p.numel() * p.element_size() for name, p in self.global_model.named_parameters() if 'fc_out' not in name)
        self.shared_model_size_kb = shared_bytes / 1024.0

    def get_shared_parameters(self):
        """Returns only shared base parameters (encoder + decoder_lstm) of global model."""
        return {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items() if 'fc_out' not in k}

    def set_shared_parameters(self, shared_params):
        """Loads shared parameters into the global model."""
        current_state = self.global_model.state_dict()
        current_state.update(shared_params)
        self.global_model.load_state_dict(current_state)

    def aggregate_shared_only(self, client_shared_weights_list, sample_counts):
        """
        Aggregates only the shared parameters using FedAvg weighting:
        w_shared_global = sum_{k=1}^K (n_k / N_total) * w_shared_k
        """
        total_samples = sum(sample_counts)
        client_shares = [n / total_samples for n in sample_counts]

        aggregated_dict = {}
        for key in client_shared_weights_list[0].keys():
            weighted_tensor = torch.zeros_like(client_shared_weights_list[0][key], dtype=torch.float32)
            for i, client_params in enumerate(client_shared_weights_list):
                weighted_tensor += client_shares[i] * client_params[key].to(torch.float32)

            if client_shared_weights_list[0][key].dtype != torch.float32:
                weighted_tensor = weighted_tensor.to(client_shared_weights_list[0][key].dtype)
            aggregated_dict[key] = weighted_tensor

        self.set_shared_parameters(aggregated_dict)
        return aggregated_dict

    def run_fedper_training(self, num_rounds=12, local_epochs=2):
        """
        Executes full FedPer personalized federated training simulation.
        Only shared parameters are communicated, tracked, and aggregated.
        Per-client evaluation uses each client's local model with its personal head.
        """
        print(f"\n=======================================================", flush=True)
        print(f"Starting Federated Learning Training (FedPer - Personalized Heads)", flush=True)
        print(f"Rounds: {num_rounds}, Local Epochs/Round: {local_epochs}, Clients: {len(self.clients)}", flush=True)
        print(f"Shared Model Size: {self.shared_model_size_kb:.2f} KB | Total Parameters: {sum(p.numel() for p in self.global_model.parameters()):,}", flush=True)
        print(f"=======================================================\n", flush=True)

        history = {
            'rounds': [],
            'train_losses': [],
            'global_ade_m': [],
            'global_fde_m': [],
            'global_rmse_m': [],
            'client_ade_m': {client.client_id: [] for client in self.clients},
            'client_fde_m': {client.client_id: [] for client in self.clients},
            'cum_comm_mb': [],
            'round_times': []
        }

        total_comm_bytes = 0

        for r in range(1, num_rounds + 1):
            t_start = time.time()
            global_shared = self.get_shared_parameters()

            client_shared_weights = []
            client_samples = []
            round_losses = []

            # 1b, 1c, 1d: Send shared params, train locally, collect updated shared params
            for client in self.clients:
                client.set_shared_parameters(copy.deepcopy(global_shared))
                total_comm_bytes += self.shared_model_size_kb * 1024

                local_loss = client.local_train(epochs=local_epochs)
                round_losses.append(local_loss)

                updated_shared = client.get_shared_parameters()
                client_shared_weights.append(updated_shared)
                client_samples.append(client.sample_count)
                total_comm_bytes += self.shared_model_size_kb * 1024

            # 1e, 1f: Aggregate shared parameters and update global model's shared parameters
            self.aggregate_shared_only(client_shared_weights, client_samples)

            # 1g: Evaluate global model on centralized test set
            global_metrics = evaluate_model_on_loader(self.global_model, self.test_loader, self.scalers, self.device)

            # 1h: Evaluate each client's LOCAL model (with personal head) on their own test set
            client_evals = {}
            for client in self.clients:
                c_metrics = evaluate_model_on_loader(client.model, client.test_loader, self.scalers, self.device)
                client_evals[client.client_id] = c_metrics
                history['client_ade_m'][client.client_id].append(c_metrics['ADE_m'])
                history['client_fde_m'][client.client_id].append(c_metrics['FDE_m'])

            elapsed = time.time() - t_start
            cum_mb = total_comm_bytes / (1024 * 1024)

            # 1i: Track all metrics, communication bytes, timing
            history['rounds'].append(r)
            history['train_losses'].append(float(np.mean(round_losses)))
            history['global_ade_m'].append(global_metrics['ADE_m'])
            history['global_fde_m'].append(global_metrics['FDE_m'])
            history['global_rmse_m'].append(global_metrics['RMSE_m'])
            history['cum_comm_mb'].append(cum_mb)
            history['round_times'].append(elapsed)

            client_ade_str = " | ".join([f"C{client.client_id}({client.name}): {client_evals[client.client_id]['ADE_m']:.3f}m" for client in self.clients])
            print(f"[Round {r:02d}/{num_rounds:02d}] "
                  f"Avg Loss: {np.mean(round_losses):.4f} | "
                  f"Global ADE: {global_metrics['ADE_m']:.3f}m ({global_metrics['ADE_ft']:.2f}ft) | "
                  f"Global FDE: {global_metrics['FDE_m']:.3f}m | "
                  f"[{client_ade_str}] | Time: {elapsed:.2f}s", flush=True)

        print(f"\n[FedPer Complete] Total Cumulative Comm: {cum_mb:.2f} MB | Final Global ADE: {history['global_ade_m'][-1]:.3f}m | Final FDE: {history['global_fde_m'][-1]:.3f}m", flush=True)
        return history


class PFedMeClient(FederatedClient):
    """
    pFedMe Client (Personalized Federated Learning with Moreau Envelopes - Dinh et al.).
    Optimizes a personalized local model using inner Moreau envelope regularization,
    and returns updated global parameters computed via bi-level gradient steps.
    """
    def __init__(self, client_id, name, train_loader, val_loader, test_loader,
                 model_class, scalers, device=DEVICE, lr=LEARNING_RATE,
                 lamda=15.0, K_inner=5, lr_inner=0.005):
        super().__init__(client_id, name, train_loader, val_loader, test_loader,
                         model_class, scalers, device, lr)
        self.personalized_model = copy.deepcopy(self.model).to(device)
        self.lamda = lamda
        self.K_inner = K_inner
        self.lr_inner = lr_inner
        self.last_loss = 0.0
        self._is_initialized = False

    def evaluate(self, split='test'):
        """Evaluates personalized model on chosen split using ADE/FDE/RMSE metrics."""
        return self.evaluate_personalized(split)

    def personalized_train(self, global_params, epochs=1, beta=0.02):
        """
        Runs pFedMe personalized training for the specified number of epochs.
        For each epoch:
          1. Set the model parameters to global_params (theta_global)
          2. For K_inner steps:
             a. Forward pass on a batch
             b. Compute task loss (MSE)
             c. Add Moreau regularization: (lamda/2) * ||theta_personal - theta_global||^2
             d. Backward + step on the personalized model
          3. After inner loop, compute global update direction:
             theta_global_new = theta_global - beta * lamda * (theta_global - theta_personal)
          4. Return the updated global parameters (for aggregation)
        """
        if not self._is_initialized:
            set_model_parameters(self.personalized_model, global_params)
            self._is_initialized = True

        current_global = copy.deepcopy(global_params)
        epoch_losses = []

        optimizer = optim.SGD(self.personalized_model.parameters(), lr=self.lr_inner)
        train_iter = iter(self.train_loader)

        for ep in range(epochs):
            # 1. Set model parameters to global_params (this is theta_global)
            self.set_parameters(current_global)
            global_param_tensors = {k: v.to(self.device) for k, v in current_global.items()}

            self.personalized_model.train()
            step_losses = []

            # 2. For K_inner steps
            for step in range(self.K_inner):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()

                predictions = self.personalized_model(inputs, targets)
                task_loss = self.criterion(predictions, targets)

                # Moreau regularization: (lamda/2) * ||theta_personal - theta_global||^2
                reg_term = 0.0
                for name, param in self.personalized_model.named_parameters():
                    if name in global_param_tensors:
                        reg_term += torch.sum((param - global_param_tensors[name]) ** 2)

                loss = task_loss + (self.lamda / 2.0) * reg_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.personalized_model.parameters(), max_norm=5.0)
                optimizer.step()

                step_losses.append(task_loss.item())

            # 3. After inner loop, compute global update direction:
            # theta_global_new = theta_global - beta * lamda * (theta_global - theta_personal)
            theta_personal = get_model_parameters(self.personalized_model)
            theta_global_new = {}
            for name, g_tensor in current_global.items():
                if name in theta_personal:
                    p_tensor = theta_personal[name].to(g_tensor.device)
                    if g_tensor.is_floating_point():
                        diff = g_tensor - p_tensor
                        step = torch.clamp(beta * self.lamda * diff, -1.0, 1.0)
                        theta_global_new[name] = g_tensor - step
                    else:
                        theta_global_new[name] = g_tensor.clone()
                else:
                    theta_global_new[name] = g_tensor.clone()

            current_global = theta_global_new
            epoch_losses.append(np.mean(step_losses) if step_losses else 0.0)

        self.set_parameters(current_global)
        self.last_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # 4. Return updated global parameters
        return current_global

    def evaluate_personalized(self, split='test'):
        """Evaluates personalized model on chosen split using ADE/FDE/RMSE metrics."""
        loader = self.test_loader if split == 'test' else self.val_loader
        return evaluate_model_on_loader(self.personalized_model, loader, self.scalers, self.device)


class PFedMeServer(FederatedServer):
    """
    pFedMe Orchestrator for Bi-level Personalized Federated Learning.
    Orchestrates global aggregation while each client personalizes via Moreau envelopes.
    """
    def __init__(self, model_class, clients, centralized_test_loader, scalers, device=DEVICE):
        super().__init__(model_class, clients, centralized_test_loader, scalers, device)

    def run_pfedme_training(self, num_rounds=12, local_epochs=1, lamda=15.0, K_inner=5, lr_inner=0.005, beta=0.02):
        """
        Executes full pFedMe training simulation.
        Broadcasts global params, runs Moreau envelope personalized_train on clients,
        aggregates updated global parameters with FedAvg, and evaluates client personalized models.
        """
        print(f"\n=======================================================", flush=True)
        print(f"Starting Federated Learning Training (pFedMe - Moreau Envelopes)", flush=True)
        print(f"Rounds: {num_rounds}, Local Epochs/Round: {local_epochs}, Clients: {len(self.clients)}", flush=True)
        print(f"Params: lamda={lamda}, K_inner={K_inner}, lr_inner={lr_inner}, beta={beta}", flush=True)
        print(f"Model Size: {self.model_size_kb:.2f} KB | Total Parameters: {sum(p.numel() for p in self.global_model.parameters()):,}", flush=True)
        print(f"=======================================================\n", flush=True)

        for client in self.clients:
            if hasattr(client, 'lamda'):
                client.lamda = lamda
            if hasattr(client, 'K_inner'):
                client.K_inner = K_inner
            if hasattr(client, 'lr_inner'):
                client.lr_inner = lr_inner

        history = {
            'rounds': [],
            'train_losses': [],
            'global_ade_m': [],
            'global_fde_m': [],
            'global_rmse_m': [],
            'client_ade_m': {client.client_id: [] for client in self.clients},
            'client_fde_m': {client.client_id: [] for client in self.clients},
            'cum_comm_mb': [],
            'round_times': []
        }

        total_comm_bytes = 0

        for r in range(1, num_rounds + 1):
            t_start = time.time()
            global_params = self.get_parameters()

            client_weights = []
            client_samples = []
            round_losses = []

            # 1a, 1b, 1c: Broadcast global params, run personalized_train, collect updated global params
            for client in self.clients:
                total_comm_bytes += self.model_size_kb * 1024
                updated_params = client.personalized_train(global_params, epochs=local_epochs, beta=beta)
                client_weights.append(updated_params)
                client_samples.append(client.sample_count)
                round_losses.append(getattr(client, 'last_loss', 0.0))
                total_comm_bytes += self.model_size_kb * 1024

            # 1d: Server Aggregation (FedAvg)
            self.aggregate_fedavg(client_weights, client_samples)

            # 1e: Global Evaluation on Centralized Test Set
            global_metrics = evaluate_model_on_loader(self.global_model, self.test_loader, self.scalers, self.device)

            # 1f: Evaluate each client's PERSONALIZED model on their own test set
            client_evals = {}
            for client in self.clients:
                if hasattr(client, 'evaluate_personalized'):
                    c_metrics = client.evaluate_personalized(split='test')
                else:
                    c_metrics = evaluate_model_on_loader(client.model, client.test_loader, self.scalers, self.device)
                client_evals[client.client_id] = c_metrics
                history['client_ade_m'][client.client_id].append(c_metrics['ADE_m'])
                history['client_fde_m'][client.client_id].append(c_metrics['FDE_m'])

            elapsed = time.time() - t_start
            cum_mb = total_comm_bytes / (1024 * 1024)

            # 1g: Track metrics, communication, timing
            history['rounds'].append(r)
            history['train_losses'].append(float(np.mean(round_losses)))
            history['global_ade_m'].append(global_metrics['ADE_m'])
            history['global_fde_m'].append(global_metrics['FDE_m'])
            history['global_rmse_m'].append(global_metrics['RMSE_m'])
            history['cum_comm_mb'].append(cum_mb)
            history['round_times'].append(elapsed)

            client_ade_str = " | ".join([f"C{client.client_id}({client.name}): {client_evals[client.client_id]['ADE_m']:.3f}m" for client in self.clients])
            print(f"[Round {r:02d}/{num_rounds:02d}] "
                  f"Avg Loss: {np.mean(round_losses):.4f} | "
                  f"Global ADE: {global_metrics['ADE_m']:.3f}m ({global_metrics['ADE_ft']:.2f}ft) | "
                  f"Global FDE: {global_metrics['FDE_m']:.3f}m | "
                  f"[{client_ade_str}] | Time: {elapsed:.2f}s", flush=True)

        print(f"\n[pFedMe Complete] Total Cumulative Comm: {cum_mb:.2f} MB | Final Global ADE: {history['global_ade_m'][-1]:.3f}m | Final FDE: {history['global_fde_m'][-1]:.3f}m", flush=True)
        return history

