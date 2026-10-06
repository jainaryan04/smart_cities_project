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
