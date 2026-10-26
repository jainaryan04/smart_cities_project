"""
Updated Master Experiment Runner with:
- Clean Contiguous Trajectory Pipeline
- Constant Velocity and Constant Acceleration Physical Baselines
- Centralized Zone-Aware and Matched Budget References
- Scale-Normalized Error (NADE)
- Renyi Differential Privacy (RDP) Moments Accountant
- 5 Repeated Seed Evaluation with Mean +/- SD and Paired Significance Tests
- Exact Test Window Direct Pooled Evaluation
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from config import (
    BASE_DIR, OUTPUT_DIR, PLOTS_DIR, DEVICE, FEET_TO_METERS,
    OBS_STEPS, PRED_STEPS, CLIENT_NAMES
)
from data_preprocessing import NGSIMPreprocessor, get_dataloaders, TrajectoryDataset
from models import (
    Seq2SeqLSTM, ConstantVelocityBaseline, ConstantAccelerationBaseline,
    ZoneAwareSeq2SeqLSTM, compute_model_size_kb, get_model_parameters
)
from metrics import (
    evaluate_model_on_loader, unscale_coordinates, compute_ade, compute_fde,
    compute_rmse, compute_mae, compute_mape, compute_nade, compute_horizon_ades,
    compute_fairness_metrics
)
from fl_framework import (
    FederatedClient, FederatedServer, CentralizedTrainer,
    CentralizedZoneAwareTrainer, IsolatedTrainer,
    FedPerClient, FedPerServer, PFedMeClient, PFedMeServer
)
from heterogeneity_metrics import (
    summarize_zone_statistics, compute_pairwise_heterogeneity,
    compute_overall_heterogeneity_score, compute_gradient_divergence,
    create_controlled_heterogeneity_partition
)
from privacy_attacks import (
    GradientReconstructionAttack, MembershipInferenceAttack,
    DifferentialPrivacyDefense
)

def run_single_seed_experiment(normalized_data, scalers, seed=42, rounds=12, cent_epochs=12, device=DEVICE):
    """Executes a full run for a given seed."""
    print(f"\n>>> Running Benchmark for Random Seed {seed} <<<", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    client_loaders, cent_loaders = get_dataloaders(normalized_data, batch_size=128)
    model_kb = compute_model_size_kb(Seq2SeqLSTM())

    # 1. Physical Baselines
    cv_model = ConstantVelocityBaseline(pred_steps=PRED_STEPS, scalers=scalers)
    ca_model = ConstantAccelerationBaseline(pred_steps=PRED_STEPS, scalers=scalers)

    # 2. Centralized Model (Standard Pooled) - Matched Budget (cent_epochs = 12)
    print(f"  [Seed {seed}] Training Centralized Pooled ({cent_epochs} epochs)...", flush=True)
    cent_trainer = CentralizedTrainer(Seq2SeqLSTM, cent_loaders, scalers, device=device)
    cent_hist, cent_model = cent_trainer.train(epochs=cent_epochs)

    # 3. Centralized Zone-Aware Reference - Matched Budget
    print(f"  [Seed {seed}] Training Centralized Zone-Aware ({cent_epochs} epochs)...", flush=True)
    cent_za_trainer = CentralizedZoneAwareTrainer(ZoneAwareSeq2SeqLSTM, client_loaders, scalers, device=device)
    cent_za_model = cent_za_trainer.train(epochs=cent_epochs)

    # 4. Isolated Silos
    print(f"  [Seed {seed}] Training Isolated Silos ({cent_epochs} epochs)...", flush=True)
    iso_trainer = IsolatedTrainer(Seq2SeqLSTM, client_loaders, scalers, device=device)
    iso_models, _ = iso_trainer.train_all(epochs=cent_epochs)

    # 5. FedAvg (Global FL) - 12 rounds
    print(f"  [Seed {seed}] Training FedAvg ({rounds} rounds)...", flush=True)
    fedavg_clients = [
        FederatedClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                        Seq2SeqLSTM, scalers, device=device)
        for cid, loaders in client_loaders.items()
    ]
    fedavg_server = FederatedServer(Seq2SeqLSTM, fedavg_clients, cent_loaders['test'], scalers, device=device)
    fedavg_hist = fedavg_server.run_federated_training(num_rounds=rounds, local_epochs=1, fedprox_mu=0.0)
    fedavg_model = fedavg_server.global_model

    # 6. FedProx (mu=0.01) - 12 rounds
    print(f"  [Seed {seed}] Training FedProx ({rounds} rounds)...", flush=True)
    fedprox_clients = [
        FederatedClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                        Seq2SeqLSTM, scalers, device=device)
        for cid, loaders in client_loaders.items()
    ]
    fedprox_server = FederatedServer(Seq2SeqLSTM, fedprox_clients, cent_loaders['test'], scalers, device=device)
    fedprox_hist = fedprox_server.run_federated_training(num_rounds=rounds, local_epochs=1, fedprox_mu=0.01)
    fedprox_model = fedprox_server.global_model

    # 7. FedPer (Personalized Heads) - 12 rounds
    print(f"  [Seed {seed}] Training FedPer ({rounds} rounds)...", flush=True)
    fedper_clients = [
        FedPerClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                     Seq2SeqLSTM, scalers, device=device)
        for cid, loaders in client_loaders.items()
    ]
    fedper_server = FedPerServer(Seq2SeqLSTM, fedper_clients, cent_loaders['test'], scalers, device=device)
    fedper_hist = fedper_server.run_fedper_training(num_rounds=rounds, local_epochs=1)
    fedper_model = fedper_server.global_model

    # 8. pFedMe (Moreau Envelopes) - 12 rounds
    print(f"  [Seed {seed}] Training pFedMe ({rounds} rounds)...", flush=True)
    pfedme_clients = [
        PFedMeClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                     Seq2SeqLSTM, scalers, device=device, lamda=1.0, K_inner=3, lr_inner=0.01)
        for cid, loaders in client_loaders.items()
    ]
    pfedme_server = PFedMeServer(Seq2SeqLSTM, pfedme_clients, cent_loaders['test'], scalers, device=device)
    pfedme_hist = pfedme_server.run_pfedme_training(num_rounds=rounds, local_epochs=1, lamda=1.0, K_inner=3, lr_inner=0.01, beta=0.2)
    pfedme_model = pfedme_server.global_model

    # Evaluate all models across zones and pooled
    results = {}
    models_to_eval = {
        'Constant Velocity': {'type': 'cv', 'model': cv_model},
        'Constant Acceleration': {'type': 'ca', 'model': ca_model},
        'Centralized (Pooled)': {'type': 'global', 'model': cent_model},
        'Centralized (Zone-Aware)': {'type': 'za', 'model': cent_za_model},
        'FedAvg (Global FL)': {'type': 'global', 'model': fedavg_model},
        'FedProx (mu=0.01)': {'type': 'global', 'model': fedprox_model},
        'FedPer (Personalized)': {'type': 'fedper', 'server': fedper_server},
        'pFedMe (Moreau)': {'type': 'pfedme', 'server': pfedme_server},
        'Isolated (US-101 Silo)': {'type': 'iso_fixed', 'model': iso_models[0]},
        'Isolated (Lankershim Silo)': {'type': 'iso_fixed', 'model': iso_models[1]},
        'Isolated (Peachtree Silo)': {'type': 'iso_fixed', 'model': iso_models[2]},
        'Isolated (Own Zone)': {'type': 'iso_own', 'models': iso_models}
    }

    zone_loaders = {
        'US-101 (Highway)': client_loaders[0]['test'],
        'Lankershim (Arterial)': client_loaders[1]['test'],
        'Peachtree (Downtown)': client_loaders[2]['test']
    }

    for m_name, m_spec in models_to_eval.items():
        results[m_name] = {}
        all_zone_preds_m = []
        all_zone_gts_m = []

        for z_name, z_loader in zone_loaders.items():
            cid = 0 if 'US-101' in z_name else (1 if 'Lankershim' in z_name else 2)
            m_type = m_spec['type']

            if m_type in ['cv', 'ca', 'global', 'iso_fixed']:
                res = evaluate_model_on_loader(m_spec['model'], z_loader, scalers, device=device)
            elif m_type == 'za':
                m_spec['model'].eval()
                # Run with zone_id
                preds_norm, gts_raw = [], []
                with torch.no_grad():
                    for batch in z_loader:
                        inp = batch[0].to(device)
                        tgt_raw = batch[2] if len(batch) == 3 else batch[1]
                        pred = m_spec['model'](inp, zone_id=cid)
                        preds_norm.append(pred.cpu().numpy())
                        gts_raw.append(tgt_raw.cpu().numpy())
                pn = np.concatenate(preds_norm, axis=0)
                gr = np.concatenate(gts_raw, axis=0)
                pf = unscale_coordinates(pn, scalers['y_mean'], scalers['y_std'])
                pm = pf * FEET_TO_METERS
                gm = gr * FEET_TO_METERS
                res = {
                    'ADE_m': compute_ade(pm, gm),
                    'FDE_m': compute_fde(pm, gm),
                    'RMSE_m': compute_rmse(pm, gm),
                    'MAE_m': compute_mae(pm, gm),
                    'MAPE_m': compute_mape(pm, gm),
                    'NADE_pct': compute_nade(pm, gm),
                    'ADE_ft': compute_ade(pf, gr),
                    'FDE_ft': compute_fde(pf, gr),
                    'RMSE_ft': compute_rmse(pf, gr),
                    'horizons_m': compute_horizon_ades(pm, gm),
                    'preds_raw_meters': pm,
                    'gts_raw_meters': gm
                }
            elif m_type == 'fedper':
                client_model = m_spec['server'].clients[cid].model
                res = evaluate_model_on_loader(client_model, z_loader, scalers, device=device)
            elif m_type == 'pfedme':
                client_model = m_spec['server'].clients[cid].personalized_model
                res = evaluate_model_on_loader(client_model, z_loader, scalers, device=device)
            elif m_type == 'iso_own':
                res = evaluate_model_on_loader(m_spec['models'][cid], z_loader, scalers, device=device)

            results[m_name][z_name] = res
            all_zone_preds_m.append(res['preds_raw_meters'])
            all_zone_gts_m.append(res['gts_raw_meters'])

        # Pooled metrics directly computed across all test predictions
        pooled_preds_m = np.concatenate(all_zone_preds_m, axis=0)
        pooled_gts_m = np.concatenate(all_zone_gts_m, axis=0)
        pooled_preds_ft = pooled_preds_m / FEET_TO_METERS
        pooled_gts_ft = pooled_gts_m / FEET_TO_METERS

        pooled_res = {
            'ADE_m': compute_ade(pooled_preds_m, pooled_gts_m),
            'FDE_m': compute_fde(pooled_preds_m, pooled_gts_m),
            'RMSE_m': compute_rmse(pooled_preds_m, pooled_gts_m),
            'MAE_m': compute_mae(pooled_preds_m, pooled_gts_m),
            'MAPE_m': compute_mape(pooled_preds_m, pooled_gts_m),
            'NADE_pct': compute_nade(pooled_preds_m, pooled_gts_m),
            'ADE_ft': compute_ade(pooled_preds_ft, pooled_gts_ft),
            'FDE_ft': compute_fde(pooled_preds_ft, pooled_gts_ft),
            'RMSE_ft': compute_rmse(pooled_preds_ft, pooled_gts_ft),
            'horizons_m': compute_horizon_ades(pooled_preds_m, pooled_gts_m),
            'preds_raw_meters': pooled_preds_m,
            'gts_raw_meters': pooled_gts_m
        }
        results[m_name]['Global Pooled'] = pooled_res

    # Communication Overhead (MiB)
    num_c = 3
    comm_mib = {
        'Constant Velocity': 0.00,
        'Constant Acceleration': 0.00,
        'Centralized (Pooled)': 15.26,
        'Centralized (Zone-Aware)': 15.26,
        'FedAvg (Global FL)': round(rounds * 2 * num_c * model_kb / 1024.0, 2),
        'FedProx (mu=0.01)': round(rounds * 2 * num_c * model_kb / 1024.0, 2),
        'FedPer (Personalized)': round(rounds * 2 * num_c * fedper_server.shared_model_size_kb / 1024.0, 2),
        'pFedMe (Moreau)': round(rounds * 2 * num_c * model_kb / 1024.0, 2),
        'Isolated (US-101 Silo)': 0.00,
        'Isolated (Lankershim Silo)': 0.00,
        'Isolated (Peachtree Silo)': 0.00,
        'Isolated (Own Zone)': 0.00
    }

    histories = {
        'cent': cent_hist,
        'fedavg': fedavg_hist,
        'fedprox': fedprox_hist,
        'fedper': fedper_hist,
        'pfedme': pfedme_hist
    }

    sample_cases = []
    # Collect sample trajectory prediction cases for visualization
    for cid in [0, 1, 2]:
        loader = client_loaders[cid]['test']
        batch = next(iter(loader))
        inp = batch[0][:1].to(device)
        raw_tg = batch[2][:1].numpy() if len(batch) == 3 else batch[1][:1].numpy()
        
        # Ground truth in meters
        gt_m = raw_tg[0] * FEET_TO_METERS
        
        # Past obs in meters
        inp_unscaled = unscale_coordinates(inp[0, :, :2].cpu().numpy(), scalers['x_mean'], scalers['x_std'])
        obs_m = inp_unscaled * FEET_TO_METERS
        
        # Predictions
        with torch.no_grad():
            f_pred = fedavg_model(inp).cpu().numpy()[0]
            c_pred = cent_model(inp).cpu().numpy()[0]
            p_pred = fedper_server.clients[cid].model(inp).cpu().numpy()[0]
            
        fed_m = unscale_coordinates(f_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS
        cent_m = unscale_coordinates(c_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS
        per_m = unscale_coordinates(p_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS

        loc_name = CLIENT_NAMES[cid]
        sample_cases.append({
            'title': f"Case {cid+1}: {loc_name}",
            'obs': obs_m,
            'gt': gt_m,
            'fedavg': fed_m,
            'cent': cent_m,
            'isolated': per_m
        })

    return results, comm_mib, histories, sample_cases


if __name__ == '__main__':
    preprocessor = NGSIMPreprocessor()
    data, scalers = preprocessor.prepare_dataset(force_recompute=False)
    print("Testing 1 seed run (seed 42, 3 rounds for fast check)...")
    res, comm, hist, cases = run_single_seed_experiment(data, scalers, seed=42, rounds=3, cent_epochs=3, device="cpu")
    print("\nTest completed successfully!")
    for m in res:
        print(f"{m:30s} -> Pooled ADE: {res[m]['Global Pooled']['ADE_m']:.3f} m, NADE: {res[m]['Global Pooled']['NADE_pct']:.2f}%")
