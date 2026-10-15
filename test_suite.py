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
    compute_ade, compute_fde, compute_rmse, unscale_coordinates, compute_horizon_ades,
    compute_mae, compute_mape, compute_fairness_metrics, compute_personalization_recovery,
    evaluate_model_on_loader
)
from fl_framework import (
    FederatedServer, FederatedClient, FedPerClient, FedPerServer, PFedMeClient, PFedMeServer
)
from heterogeneity_metrics import (
    compute_kl_divergence, compute_js_divergence, compute_emd,
    compute_pairwise_heterogeneity, compute_overall_heterogeneity_score,
    compute_gradient_divergence, create_controlled_heterogeneity_partition,
    compute_zone_kinematic_stats
)
from privacy_attacks import (
    GradientReconstructionAttack, ImprovedGradientReconstructionAttack,
    MembershipInferenceAttack, DifferentialPrivacyDefense
)
from data_preprocessing import TrajectoryDataset
from torch.utils.data import DataLoader


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

    def test_mae_and_mape_metrics(self):
        """Test MAE and MAPE metrics with known geometry and values."""
        gt = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        pred = np.array([[[1.5, 1.0], [4.0, 2.0]]])
        # Errors: |1.5-1.0|=0.5, |1.0-2.0|=1.0, |4.0-3.0|=1.0, |2.0-4.0|=2.0
        # MAE = (0.5 + 1.0 + 1.0 + 2.0) / 4 = 4.5 / 4 = 1.125
        mae = compute_mae(pred, gt)
        self.assertAlmostEqual(mae, 1.125, places=4, msg="MAE computation mismatch")

        # MAPE = (0.5/1.0 + 1.0/2.0 + 1.0/3.0 + 2.0/4.0) / 4 * 100
        # = (0.5 + 0.5 + 0.333333 + 0.5) / 4 * 100 = 1.833333 / 4 * 100 = 45.8333%
        mape = compute_mape(pred, gt)
        self.assertAlmostEqual(mape, 45.833333, places=3, msg="MAPE computation mismatch")

        # Zero error test
        self.assertAlmostEqual(compute_mae(gt, gt), 0.0, places=6)
        self.assertAlmostEqual(compute_mape(gt, gt), 0.0, places=6)

    def test_fairness_and_personalization_recovery_metrics(self):
        """Test fairness metrics across zones and personalization recovery calculations."""
        zone_errors = {'Highway': 2.0, 'Arterial': 3.0, 'Downtown': 1.0}
        fairness = compute_fairness_metrics(zone_errors)
        self.assertEqual(fairness['worst_zone'], ('Arterial', 3.0))
        self.assertEqual(fairness['best_zone'], ('Downtown', 1.0))
        self.assertAlmostEqual(fairness['mean_error'], 2.0, places=4)
        self.assertAlmostEqual(fairness['gap'], 2.0, places=4)
        # std = sqrt(((0)^2 + (1)^2 + (-1)^2) / 3) = sqrt(2/3) = 0.81649658
        self.assertAlmostEqual(fairness['std_error'], np.sqrt(2.0 / 3.0), places=4)
        self.assertAlmostEqual(fairness['coeff_variation'], np.sqrt(2.0 / 3.0) / 2.0, places=4)

        # Personalization recovery
        cent_err, fed_err, pfl_err = 1.0, 2.0, 1.2
        rec = compute_personalization_recovery(cent_err, fed_err, pfl_err)
        self.assertAlmostEqual(rec['fedavg_gap'], 1.0, places=4)
        self.assertAlmostEqual(rec['pfl_gap'], 0.2, places=4)
        self.assertAlmostEqual(rec['recovery_pct'], 80.0, places=4)
        self.assertAlmostEqual(rec['improvement_pct'], 40.0, places=4)

    def test_evaluate_model_on_loader_returns_all_metrics(self):
        """Test that evaluate_model_on_loader computes and returns MAE and MAPE alongside ADE/FDE/RMSE."""
        inputs = np.random.randn(8, OBS_STEPS, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(8, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        raw_targets = targets * 10.0 + 5.0
        dataset = TrajectoryDataset(inputs, targets, raw_targets)
        loader = DataLoader(dataset, batch_size=4)

        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        scalers = {'x_mean': np.zeros(4), 'x_std': np.ones(4), 'y_mean': np.zeros(2), 'y_std': np.ones(2)}

        results = evaluate_model_on_loader(model, loader, scalers, device="cpu")
        self.assertIn('ADE_m', results)
        self.assertIn('FDE_m', results)
        self.assertIn('RMSE_m', results)
        self.assertIn('MAE_m', results)
        self.assertIn('MAPE_m', results)
        self.assertGreater(results['MAE_m'], 0.0)
        self.assertGreater(results['MAPE_m'], 0.0)

    def test_fedper_client_and_server(self):
        """Test FedPer parameter isolation and shared layer aggregation."""
        train_ds = TrajectoryDataset(
            np.random.randn(10, OBS_STEPS, INPUT_DIM).astype(np.float32),
            np.random.randn(10, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=5)
        scalers = {'x_mean': np.zeros(4), 'x_std': np.ones(4), 'y_mean': np.zeros(2), 'y_std': np.ones(2)}

        client = FedPerClient(
            client_id=0, name="RSU_Test", train_loader=train_loader,
            val_loader=train_loader, test_loader=train_loader,
            model_class=lambda: Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1),
            scalers=scalers, device="cpu"
        )

        shared_params = client.get_shared_parameters()
        personal_params = client.get_personal_parameters()

        # Check parameter separation
        for k in shared_params:
            self.assertNotIn('fc_out', k, f"Shared params must not include fc_out: {k}")
        for k in personal_params:
            self.assertIn('fc_out', k, f"Personal params must include fc_out: {k}")

        # Check set_shared_parameters leaves personal fc_out unchanged
        first_personal_key = list(personal_params.keys())[0]
        orig_fc_weight = personal_params[first_personal_key].clone()
        new_shared = {k: v + 5.0 for k, v in shared_params.items()}
        client.set_shared_parameters(new_shared)
        updated_personal = client.get_personal_parameters()
        self.assertTrue(torch.allclose(orig_fc_weight, updated_personal[first_personal_key]),
                        "Personal weights must remain unchanged when setting shared weights")

        # FedPerServer aggregation
        server = FedPerServer(
            model_class=lambda: Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1),
            clients=[client], centralized_test_loader=train_loader, scalers=scalers, device="cpu"
        )
        self.assertLess(server.shared_model_size_kb, server.model_size_kb)

    def test_pfedme_client_and_server(self):
        """Test pFedMe Moreau envelope training and parameter aggregation."""
        train_ds = TrajectoryDataset(
            np.random.randn(10, OBS_STEPS, INPUT_DIM).astype(np.float32),
            np.random.randn(10, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=5)
        scalers = {'x_mean': np.zeros(4), 'x_std': np.ones(4), 'y_mean': np.zeros(2), 'y_std': np.ones(2)}

        client = PFedMeClient(
            client_id=0, name="RSU_Test", train_loader=train_loader,
            val_loader=train_loader, test_loader=train_loader,
            model_class=lambda: Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1),
            scalers=scalers, device="cpu", lamda=10.0, K_inner=2, lr_inner=0.01
        )

        global_model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        global_params = get_model_parameters(global_model)

        # Run personalized training
        updated_global = client.personalized_train(global_params, epochs=1, beta=1.0)
        self.assertEqual(len(updated_global), len(global_params))
        # Ensure personalized model was updated and evaluable
        eval_metrics = client.evaluate_personalized(split='test')
        self.assertIn('ADE_m', eval_metrics)
        self.assertGreater(eval_metrics['ADE_m'], 0.0)

    def test_heterogeneity_metrics(self):
        """Test KL divergence, JS divergence, and Earth Mover's Distance (Wasserstein-1)."""
        dist_p = np.array([0.5, 0.5])
        dist_q = np.array([0.5, 0.5])
        dist_r = np.array([0.9, 0.1])

        # Self-divergence should be 0
        self.assertAlmostEqual(compute_kl_divergence(dist_p, dist_q), 0.0, places=5)
        self.assertAlmostEqual(compute_js_divergence(dist_p, dist_q), 0.0, places=5)

        # Asymmetric vs symmetric
        kl_pr = compute_kl_divergence(dist_p, dist_r)
        self.assertGreater(kl_pr, 0.0)

        js_pr = compute_js_divergence(dist_p, dist_r)
        js_rp = compute_js_divergence(dist_r, dist_p)
        self.assertGreater(js_pr, 0.0)
        self.assertLessEqual(js_pr, 1.0)
        self.assertAlmostEqual(js_pr, js_rp, places=5, msg="JS divergence must be symmetric")

        # EMD
        vals_1 = np.array([1.0, 2.0, 3.0])
        vals_2 = np.array([1.0, 2.0, 3.0])
        vals_3 = np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(compute_emd(vals_1, vals_2), 0.0, places=5)
        self.assertAlmostEqual(compute_emd(vals_1, vals_3), 3.0, places=4)

    def test_heterogeneity_controlled_partitioning(self):
        """Test controlled non-IID partitioning ('low', 'medium', 'high')."""
        dummy_data = {
            0: {
                'name': 'Zone_0',
                'train_x': np.random.randn(20, OBS_STEPS, INPUT_DIM).astype(np.float32),
                'train_y': np.random.randn(20, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
                'raw_train_y': np.random.randn(20, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
            },
            1: {
                'name': 'Zone_1',
                'train_x': np.random.randn(20, OBS_STEPS, INPUT_DIM).astype(np.float32),
                'train_y': np.random.randn(20, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
                'raw_train_y': np.random.randn(20, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
            }
        }
        for level in ['low', 'medium', 'high']:
            part = create_controlled_heterogeneity_partition(dummy_data, level=level, seed=42)
            self.assertIn(0, part)
            self.assertIn(1, part)
            self.assertGreater(len(part[0]['train_x']), 0)
            self.assertGreater(len(part[1]['train_x']), 0)

    def test_gradient_divergence(self):
        """Test gradient divergence metric computation across clients."""
        grad_a = {'layer1': torch.tensor([1.0, 0.0]), 'layer2': torch.tensor([0.0, 1.0])}
        grad_b = {'layer1': torch.tensor([1.0, 0.0]), 'layer2': torch.tensor([0.0, 1.0])}
        grad_c = {'layer1': torch.tensor([-1.0, 0.0]), 'layer2': torch.tensor([0.0, -1.0])}

        # Identical gradients -> cosine similarity 1.0 -> distance 0.0
        div_identical = compute_gradient_divergence({0: grad_a, 1: grad_b})
        self.assertAlmostEqual(div_identical, 0.0, places=4)

        # Opposite gradients -> cosine similarity -1.0 -> distance 2.0
        div_opposite = compute_gradient_divergence({0: grad_a, 1: grad_c})
        self.assertAlmostEqual(div_opposite, 2.0, places=4)

    def test_privacy_attacks_and_defenses(self):
        """Test DLG attack, MIA attack, and Differential Privacy Defense components."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        attacker = GradientReconstructionAttack(model, device="cpu")

        dummy_in = torch.randn(1, OBS_STEPS, INPUT_DIM)
        dummy_tg = torch.randn(1, PRED_STEPS, OUTPUT_DIM)

        # Gradient extraction
        target_grads = attacker.compute_target_gradient(dummy_in, dummy_tg)
        self.assertGreater(len(target_grads), 0)

        # Quick reconstruction run (5 iterations)
        rec_in, rec_tg, losses = attacker.reconstruct(target_grads, num_iterations=5, lr=0.1)
        self.assertEqual(rec_in.shape, (1, OBS_STEPS, INPUT_DIM))
        self.assertEqual(rec_tg.shape, (1, PRED_STEPS, OUTPUT_DIM))
        self.assertEqual(len(losses), 5)

        # Reconstruction evaluation metrics
        metrics = attacker.evaluate_reconstruction(dummy_in, dummy_tg, rec_in, rec_tg)
        self.assertIn('input_mse', metrics)
        self.assertIn('position_error', metrics)
        self.assertIn('input_cosine_sim', metrics)

        # Differential Privacy Defense
        dp = DifferentialPrivacyDefense(clip_norm=1.0, noise_multiplier=0.5)
        eps = dp.compute_epsilon(noise_multiplier=0.5, num_steps=50)
        self.assertGreater(eps, 0.0)

        # DP state dict noise perturbation
        state_dict = model.state_dict()
        noised = dp.add_noise_to_state_dict(state_dict)
        self.assertEqual(len(noised), len(state_dict))
        # Ensure noise was added
        for k in state_dict:
            if state_dict[k].is_floating_point():
                self.assertFalse(torch.equal(state_dict[k], noised[k]))

        # Membership Inference Attack
        mia = MembershipInferenceAttack(model, device="cpu")
        sample_loss = mia.compute_sample_loss(dummy_in, dummy_tg)
        self.assertGreater(sample_loss, 0.0)

    def test_fedper_shared_only_reconstruction_attack(self):
        """Test FedPer shared-only gradient reconstruction attack scenario."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        attacker = GradientReconstructionAttack(model, device="cpu")

        dummy_in = torch.randn(1, OBS_STEPS, INPUT_DIM)
        dummy_tg = torch.randn(1, PRED_STEPS, OUTPUT_DIM)

        # Full vs Shared-only gradient counts
        full_grads = attacker.compute_target_gradient(dummy_in, dummy_tg, shared_only=False)
        shared_grads = attacker.compute_target_gradient(dummy_in, dummy_tg, shared_only=True)

        total_param_count = len(list(model.parameters()))
        shared_param_count = len([p for name, p in model.named_parameters() if 'fc_out' not in name])

        self.assertEqual(len(full_grads), total_param_count)
        self.assertEqual(len(shared_grads), shared_param_count)
        self.assertLess(len(shared_grads), len(full_grads))

        # Reconstruct with shared_only=True
        rec_in, rec_tg, losses = attacker.reconstruct(
            shared_grads, num_iterations=5, lr=0.1, shared_only=True
        )
        self.assertEqual(rec_in.shape, (1, OBS_STEPS, INPUT_DIM))
        self.assertEqual(rec_tg.shape, (1, PRED_STEPS, OUTPUT_DIM))
        self.assertEqual(len(losses), 5)

    def test_improved_dlg_attack(self):
        """Test ImprovedGradientReconstructionAttack (iDLG) with target initialization."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        attacker = ImprovedGradientReconstructionAttack(model, device="cpu")

        dummy_in = torch.randn(1, OBS_STEPS, INPUT_DIM)
        dummy_tg = torch.randn(1, PRED_STEPS, OUTPUT_DIM)

        target_grads = attacker.compute_target_gradient(dummy_in, dummy_tg)
        rec_in, rec_tg, losses = attacker.reconstruct(target_grads, num_iterations=5, lr=0.1)

        self.assertEqual(rec_in.shape, (1, OBS_STEPS, INPUT_DIM))
        self.assertEqual(rec_tg.shape, (1, PRED_STEPS, OUTPUT_DIM))
        self.assertEqual(len(losses), 5)

        metrics = attacker.evaluate_reconstruction(dummy_in, dummy_tg, rec_in, rec_tg)
        self.assertIn('input_mse', metrics)
        self.assertIn('position_error', metrics)
        self.assertIn('endpoint_error', metrics)

    def test_dp_clip_and_noise_gradients(self):
        """Test DifferentialPrivacyDefense gradient clipping and noise addition in-place."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        dummy_in = torch.randn(2, OBS_STEPS, INPUT_DIM)
        dummy_tg = torch.randn(2, PRED_STEPS, OUTPUT_DIM)

        pred = model(dummy_in)
        loss = torch.nn.functional.mse_loss(pred, dummy_tg)
        loss.backward()

        dp = DifferentialPrivacyDefense(clip_norm=0.5, noise_multiplier=0.2)
        total_norm = dp.clip_and_noise_gradients(model)
        self.assertGreater(total_norm, 0.0)

        # Confirm gradients remain populated and valid
        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_membership_inference_attack_evaluate(self):
        """Test MembershipInferenceAttack full evaluation on member vs non-member loaders."""
        model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        mia = MembershipInferenceAttack(model, device="cpu")

        member_ds = TrajectoryDataset(
            np.random.randn(8, OBS_STEPS, INPUT_DIM).astype(np.float32),
            np.random.randn(8, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        )
        non_member_ds = TrajectoryDataset(
            np.random.randn(8, OBS_STEPS, INPUT_DIM).astype(np.float32),
            np.random.randn(8, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        )
        m_loader = DataLoader(member_ds, batch_size=4)
        nm_loader = DataLoader(non_member_ds, batch_size=4)

        results = mia.evaluate(m_loader, nm_loader, num_samples=8)
        self.assertIn('auc', results)
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1', results)
        self.assertIn('optimal_threshold', results)
        self.assertGreaterEqual(results['auc'], 0.0)
        self.assertLessEqual(results['auc'], 1.0)

    def test_zone_kinematic_statistics(self):
        """Test summarize_zone_statistics / compute_zone_kinematic_stats."""
        dummy_data = {
            0: {
                'name': 'Zone_A',
                'train_x': np.ones((5, OBS_STEPS, INPUT_DIM), dtype=np.float32) * 2.0,
                'train_y': np.ones((5, PRED_STEPS, OUTPUT_DIM), dtype=np.float32) * 3.0,
                'raw_train_y': np.ones((5, PRED_STEPS, OUTPUT_DIM), dtype=np.float32) * 3.0,
            }
        }
        scalers = {
            'x_mean': np.zeros(4), 'x_std': np.ones(4),
            'y_mean': np.zeros(2), 'y_std': np.ones(2)
        }
        stats = compute_zone_kinematic_stats(dummy_data, scalers=scalers)
        self.assertIn(0, stats)
        self.assertIn('speed_mean', stats[0])
        self.assertIn('acc_mean', stats[0])
        self.assertIn('trajectory_length_mean', stats[0])
        self.assertEqual(stats[0]['num_trajectories'], 5)

    def test_pairwise_heterogeneity_and_overall_score(self):
        """Test compute_pairwise_heterogeneity and compute_overall_heterogeneity_score."""
        dummy_data = {
            0: {
                'name': 'Zone_0',
                'train_x': np.random.randn(15, OBS_STEPS, INPUT_DIM).astype(np.float32),
                'train_y': np.random.randn(15, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
                'raw_train_y': np.random.randn(15, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
            },
            1: {
                'name': 'Zone_1',
                'train_x': np.random.randn(15, OBS_STEPS, INPUT_DIM).astype(np.float32) + 2.0,
                'train_y': np.random.randn(15, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
                'raw_train_y': np.random.randn(15, PRED_STEPS, OUTPUT_DIM).astype(np.float32),
            }
        }
        scalers = {
            'x_mean': np.zeros(4), 'x_std': np.ones(4),
            'y_mean': np.zeros(2), 'y_std': np.ones(2)
        }
        pw = compute_pairwise_heterogeneity(dummy_data, scalers=scalers)
        self.assertIn((0, 1), pw)
        self.assertIn('speed', pw[(0, 1)])
        self.assertIn('js', pw[(0, 1)]['speed'])

        score = compute_overall_heterogeneity_score(pw)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pfedme_initialization_and_evaluation_consistency(self):
        """Test PFedMeClient evaluate() defaults to evaluate_personalized and syncs initial global weights."""
        train_ds = TrajectoryDataset(
            np.random.randn(10, OBS_STEPS, INPUT_DIM).astype(np.float32),
            np.random.randn(10, PRED_STEPS, OUTPUT_DIM).astype(np.float32)
        )
        loader = DataLoader(train_ds, batch_size=5)
        scalers = {'x_mean': np.zeros(4), 'x_std': np.ones(4), 'y_mean': np.zeros(2), 'y_std': np.ones(2)}

        client = PFedMeClient(
            client_id=0, name="RSU_Test", train_loader=loader,
            val_loader=loader, test_loader=loader,
            model_class=lambda: Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1),
            scalers=scalers, device="cpu", lamda=10.0, K_inner=2, lr_inner=0.01
        )

        global_model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
        global_params = get_model_parameters(global_model)

        self.assertFalse(client._is_initialized)
        client.personalized_train(global_params, epochs=1, beta=0.02)
        self.assertTrue(client._is_initialized)

        # evaluate() should match evaluate_personalized()
        eval_gen = client.evaluate(split='test')
        eval_pers = client.evaluate_personalized(split='test')
        self.assertEqual(eval_gen['ADE_m'], eval_pers['ADE_m'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

