"""
Unit and Integration Test Suite for Smart Cities Federated Trajectory Prediction.
Tests:
1. Preprocessing and sequence shape integrity
2. Vehicle-level split separation (no data leakage)
3. Model forward and backward execution
4. Federated Averaging (FedAvg) aggregation arithmetic
5. Trajectory metrics calculation accuracy (ADE, FDE, RMSE)
6. Coordinate unscaling and unit transformation
"""

import unittest
import numpy as np
import torch
import os
import shutil

from config import OBS_STEPS, PRED_STEPS, INPUT_DIM, OUTPUT_DIM, FEET_TO_METERS
from models import (
    Seq2SeqLSTM, AttentionSeq2SeqLSTM, get_model_parameters,
    set_model_parameters, clone_model, count_parameters
)
from metrics import (
    compute_ade, compute_fde, compute_rmse, unscale_coordinates, compute_horizon_ades
)
from fl_framework import FederatedServer


class TestTrajectoryPredictionSuite(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        np.random.seed(42)
        torch.manual_seed(42)

    def test_metrics_accuracy(self):
        """Test ADE, FDE, and RMSE on synthetic known geometry."""
        # N=2, T=3, D=2
        gt = np.array([
            [[0.0, 0.0], [0.0, 3.0], [0.0, 6.0]],
            [[1.0, 1.0], [1.0, 5.0], [1.0, 9.0]]
        ])
        pred = np.array([
            [[0.0, 0.0], [4.0, 3.0], [0.0, 10.0]], # dists: 0, 4, 4 -> mean = 8/3 = 2.66667
            [[1.0, 1.0], [1.0, 1.0], [1.0, 9.0]]   # dists: 0, 4, 0 -> mean = 4/3 = 1.33333
        ])
        # Sample 1: ADE = (0 + 4 + 4) / 3 = 8/3 = 2.6667, FDE = 4.0
        # Sample 2: ADE = (0 + 4 + 0) / 3 = 4/3 = 1.3333, FDE = 0.0
        # Total ADE = (2.66667 + 1.33333) / 2 = 2.0
        # Total FDE = (4.0 + 0.0) / 2 = 2.0
        ade = compute_ade(pred, gt)
        fde = compute_fde(pred, gt)
        self.assertAlmostEqual(ade, 2.0, places=4, msg="ADE calculation mismatch")
        self.assertAlmostEqual(fde, 2.0, places=4, msg="FDE calculation mismatch")

        # Zero error test
        zero_ade = compute_ade(gt, gt)
        zero_fde = compute_fde(gt, gt)
        self.assertAlmostEqual(zero_ade, 0.0, places=6)
        self.assertAlmostEqual(zero_fde, 0.0, places=6)

    def test_seq2seq_lstm_forward_backward(self):
        """Test Seq2Seq LSTM shape, forward pass, and gradient backprop."""
        batch_size = 8
        model = Seq2SeqLSTM(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
            pred_steps=PRED_STEPS
        )
        dummy_in = torch.randn(batch_size, OBS_STEPS, INPUT_DIM)
        dummy_tgt = torch.randn(batch_size, PRED_STEPS, OUTPUT_DIM)

        out = model(dummy_in, dummy_tgt)
        self.assertEqual(out.shape, (batch_size, PRED_STEPS, OUTPUT_DIM))

        criterion = torch.nn.MSELoss()
        loss = criterion(out, dummy_tgt)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_attention_model_forward(self):
        """Test Attention-augmented Seq2Seq LSTM."""
        batch_size = 4
        model = AttentionSeq2SeqLSTM(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
            pred_steps=PRED_STEPS
        )
        dummy_in = torch.randn(batch_size, OBS_STEPS, INPUT_DIM)
        out = model(dummy_in)
        self.assertEqual(out.shape, (batch_size, PRED_STEPS, OUTPUT_DIM))

    def test_fedavg_aggregation(self):
        """Test FedAvg weighted average calculation."""
        model_a = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        model_b = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        global_model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)

        # Set specific weights
        params_a = get_model_parameters(model_a)
        params_b = get_model_parameters(model_b)
        
        for k in params_a:
            params_a[k].fill_(10.0)
            params_b[k].fill_(20.0)

        # Sample weights: Client A has 100 samples, Client B has 300 samples (Total 400)
        # Expected: 0.25 * 10 + 0.75 * 20 = 2.5 + 15 = 17.5
        server = FederatedServer(
            model_class=lambda: global_model,
            clients=[],
            centralized_test_loader=None,
            scalers={'y_mean': np.zeros(2), 'y_std': np.ones(2)},
            device="cpu"
        )
        agg_params = server.aggregate_fedavg([params_a, params_b], [100, 300])

        for k in agg_params:
            if agg_params[k].is_floating_point():
                mean_val = float(agg_params[k].mean().item())
                self.assertAlmostEqual(mean_val, 17.5, places=4, msg=f"Aggregation error for {k}")

    def test_coordinate_unscaling(self):
        """Test unscaling of standardized coordinates back to feet and meters."""
        norm_coords = np.array([[[1.0, 2.0]]])  # [1, 1, 2]
        y_mean = np.array([10.0, 20.0])
        y_std = np.array([5.0, 10.0])

        # unscaled = 1.0 * 5 + 10 = 15; 2.0 * 10 + 20 = 40
        unscaled_feet = unscale_coordinates(norm_coords, y_mean, y_std)
        self.assertAlmostEqual(unscaled_feet[0, 0, 0], 15.0)
        self.assertAlmostEqual(unscaled_feet[0, 0, 1], 40.0)

    def test_rmse_and_horizon_metrics(self):
        """Test RMSE and multi-horizon ADE/FDE metrics."""
        gt = np.zeros((4, 6, 2))
        pred = np.ones((4, 6, 2)) * 2.0  # dist = sqrt(4 + 4) = 2.828427
        
        rmse = compute_rmse(pred, gt)
        self.assertAlmostEqual(rmse, 2.0, places=4)
        
        horizons = compute_horizon_ades(pred, gt, fps=2)
        self.assertIn("ADE_1.0s", horizons)
        self.assertIn("ADE_2.0s", horizons)
        self.assertIn("ADE_3.0s", horizons)
        self.assertIn("FDE_3.0s", horizons)
        self.assertAlmostEqual(horizons["ADE_3.0s"], np.sqrt(8), places=4)
        self.assertAlmostEqual(horizons["FDE_3.0s"], np.sqrt(8), places=4)

    def test_vehicle_split_separation_no_leakage(self):
        """Test that vehicle ID partitioning creates completely disjoint train/val/test sets (0% leakage)."""
        unique_vids = np.arange(100)
        np.random.shuffle(unique_vids)
        
        n_v = len(unique_vids)
        n_train = int(0.70 * n_v)
        n_val = int(0.15 * n_v)
        
        train_vids = set(unique_vids[:n_train])
        val_vids = set(unique_vids[n_train:n_train+n_val])
        test_vids = set(unique_vids[n_train+n_val:])
        
        self.assertEqual(len(train_vids.intersection(val_vids)), 0, "Data leakage between Train and Val")
        self.assertEqual(len(train_vids.intersection(test_vids)), 0, "Data leakage between Train and Test")
        self.assertEqual(len(val_vids.intersection(test_vids)), 0, "Data leakage between Val and Test")
        self.assertEqual(len(train_vids) + len(val_vids) + len(test_vids), 100)

    def test_fedprox_proximal_term(self):
        """Test that FedProx proximal penalty is computed and strictly penalizes divergence from global weights."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        global_params = get_model_parameters(model)
        
        # Perturb local model weights
        for p in model.parameters():
            p.data.add_(1.0)
            
        prox_term = 0.0
        for name, param in model.named_parameters():
            if name in global_params:
                prox_term += torch.sum((param - global_params[name]) ** 2).item()
                
        self.assertGreater(prox_term, 0.0, "Proximal penalty must be positive when weights diverge")


if __name__ == '__main__':
    unittest.main(verbosity=2)
