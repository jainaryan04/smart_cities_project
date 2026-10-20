"""
Master Journal Experiment Runner for Smart Cities Federated Vehicle Trajectory Prediction.
Coordinates and executes the full experimental matrix across:
1. Heterogeneity Quantification Study (KL, JS, EMD per zone pair and feature; gradient divergence)
2. Benchmark (Centralized, Centralized with zone heads, Isolated silos scored in and out of zone,
   own-zone silo baseline, constant-velocity / constant-acceleration extrapolation, FedAvg,
   FedProx, FedPer, pFedMe) repeated over several seeds with equal training budgets
3. Statistical tests (paired across seeds and paired over test windows)
4. Controlled Heterogeneity Analysis (Low, Medium, High non-IID training partitions)
5. Privacy: DLG against the trained models, MIA on every trained model, and DP-FedAvg trained
   with DP-SGD and an RDP accountant
6. Integrated trade-off table built only from measured values
7. Publication Results Export (CSV and Markdown Summary Report)
"""

import os
import time
import random
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from scipy import stats

from config import (
    BASE_DIR, OUTPUT_DIR, DEVICE, COMMUNICATION_ROUNDS, LOCAL_EPOCHS,
    FEDPROX_MU, FEET_TO_METERS, OBS_STEPS, PRED_STEPS, INPUT_DIM, OUTPUT_DIM,
    DEFAULT_SEEDS, RANDOM_SEED
)
from data_preprocessing import NGSIMPreprocessor, get_dataloaders
from models import Seq2SeqLSTM, compute_model_size_kb
from metrics import evaluate_model_on_loader, compute_metrics_from_predictions, compute_fairness_metrics
from fl_framework import (
    FederatedClient, FederatedServer, CentralizedTrainer, IsolatedTrainer,
    FedPerClient, FedPerServer, PFedMeClient, PFedMeServer
)
from baselines import KinematicExtrapolation, CentralizedZoneHeadTrainer
from dp_training import DPFederatedClient
from heterogeneity_metrics import (
    compute_pairwise_heterogeneity, compute_overall_heterogeneity_score,
    compute_gradient_divergence, create_controlled_heterogeneity_partition,
    summarize_zone_statistics, print_heterogeneity_report
)
from privacy_attacks import run_privacy_evaluation, run_membership_inference
from visualize import (
    plot_learning_curves,
    plot_heterogeneity_vs_personalization,
    plot_privacy_leakage_vs_dp,
    plot_fairness_disparity_comparison,
    plot_accuracy_privacy_tradeoff_pareto
)

ZONES = {0: 'US-101 (Highway)', 1: 'Lankershim (Arterial)', 2: 'Peachtree (Downtown)'}
POOLED = 'Global Pooled'
OWN_ZONE = 'Isolated (own zone)'
SILO_NAMES = {0: 'Isolated (US-101 Silo)', 1: 'Isolated (Lankershim Silo)', 2: 'Isolated (Peachtree Silo)'}


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Cities Federated Trajectory Prediction Journal Runner")
    parser.add_argument("--rounds", type=int, default=COMMUNICATION_ROUNDS, help="Federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS, help="Local client epochs per round")
    parser.add_argument("--cent-epochs", type=int, default=None,
                        help="Centralized / isolated epochs (default: rounds x local epochs, i.e. an equal budget)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds for repeated runs")
    parser.add_argument("--fedprox-mu", type=float, default=FEDPROX_MU)
    parser.add_argument("--pfedme-lambda", type=float, default=1.0)
    parser.add_argument("--tune", action="store_true",
                        help="Select FedProx mu and pFedMe lambda on the validation split before the main runs")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Computation device (cpu or cuda)")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode for rapid verification")
    parser.add_argument("--attack-samples", type=int, default=10, help="Number of attacked batches for DLG")
    parser.add_argument("--attack-batch-size", type=int, default=1, help="Windows per attacked gradient")
    parser.add_argument("--dp-sigmas", type=float, nargs="*", default=[0.5, 1.0, 2.0],
                        help="Noise multipliers for DP-FedAvg training (empty to skip)")
    parser.add_argument("--dp-clip", type=float, default=1.0, help="Per-example clipping norm C for DP-SGD")
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild the dataset cache")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================================================================
# 1. HETEROGENEITY QUANTIFICATION STUDY
# ==============================================================================

def run_heterogeneity_study(normalized_data, scalers, device="cpu"):
    """
    Executes heterogeneity quantification:
    - Zone kinematic profiles (speed, acceleration, displacement)
    - Natural geographic pairwise divergences (KL, JS, EMD), saved per pair and feature
    - Controlled non-IID partition divergence comparisons ('low', 'medium', 'high')
    - Model gradient divergence across client zones (one batch, freshly initialized model)
    """
    print("\n" + "=" * 85, flush=True)
    print("PHASE 1: GEOGRAPHIC NON-IID HETEROGENEITY QUANTIFICATION", flush=True)
    print("=" * 85, flush=True)

    zone_stats = summarize_zone_statistics(normalized_data, scalers=scalers)

    natural_pairwise = compute_pairwise_heterogeneity(normalized_data, scalers=scalers)
    natural_score = compute_overall_heterogeneity_score(natural_pairwise)
    print_heterogeneity_report(natural_pairwise, zone_stats, natural_score)

    print("[Heterogeneity] Evaluating controlled non-IID partitions ('low', 'medium', 'high')...", flush=True)
    partition_scores = {}
    pairwise_records = []

    for level in ['low', 'medium', 'high']:
        part_data = create_controlled_heterogeneity_partition(normalized_data, level=level, seed=RANDOM_SEED)
        pw_metrics = natural_pairwise if level == 'medium' else compute_pairwise_heterogeneity(part_data, scalers=scalers)
        score = compute_overall_heterogeneity_score(pw_metrics)
        partition_scores[level] = score

        for (cid_i, cid_j), feats in pw_metrics.items():
            for feat_name, vals in feats.items():
                pairwise_records.append({
                    'Partition Level': level.capitalize(),
                    'Client Pair': f"{ZONES.get(cid_i, cid_i)} vs {ZONES.get(cid_j, cid_j)}",
                    'Feature': feat_name,
                    'KL Divergence': round(vals.get('kl', 0.0), 4),
                    'JS Divergence': round(vals.get('js', 0.0), 4),
                    'EMD (Wasserstein-1)': round(vals.get('emd', 0.0), 4),
                    'Aggregate Level JS Score': round(score, 4)
                })

        print(f"  -> Partition Level '{level.upper()}': Aggregate JS Score = {score:.4f}", flush=True)

    print("\n[Heterogeneity] Computing inter-client gradient divergence (freshly initialized model)...", flush=True)
    test_model = Seq2SeqLSTM(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=64, num_layers=2).to(device)
    criterion = torch.nn.MSELoss()
    client_grads = {}

    client_loaders, _ = get_dataloaders(normalized_data, batch_size=64)
    for cid in sorted(client_loaders.keys()):
        batch = next(iter(client_loaders[cid]['train']))
        inp, tgt = batch[0].to(device), batch[1].to(device)
        test_model.zero_grad()
        loss = criterion(test_model(inp, tgt), tgt)
        loss.backward()
        client_grads[cid] = {name: p.grad.clone() for name, p in test_model.named_parameters() if p.grad is not None}

    grad_div = compute_gradient_divergence(client_grads)
    print(f"  -> Mean Pairwise Gradient Cosine Distance: {grad_div:.4f} (Range [0, 2])", flush=True)

    return {
        'zone_stats': zone_stats,
        'natural_score': natural_score,
        'partition_scores': partition_scores,
        'pairwise_records': pairwise_records,
        'gradient_divergence': grad_div
    }


# ==============================================================================
# 2. BENCHMARK TRAINING AND EVALUATION
# ==============================================================================

def predict(model, loader, device="cpu"):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch[0].to(device)).cpu().numpy())
            gts.append(batch[2].numpy() if len(batch) == 3 else batch[1].numpy())
    return np.concatenate(preds), np.concatenate(gts)


def evaluate_zone_models(zone_models, client_loaders, scalers, device="cpu"):
    """
    Evaluates a set of per-zone models: zone k's test windows are predicted by zone_models[k].
    Returns per-zone metrics and the pooled metrics over all test windows (exact window counts).
    """
    per_zone, all_p, all_g = {}, [], []
    for cid in sorted(client_loaders.keys()):
        p, g = predict(zone_models[cid], client_loaders[cid]['test'], device)
        per_zone[cid] = compute_metrics_from_predictions(p, g, scalers)
        all_p.append(p)
        all_g.append(g)
    pooled = compute_metrics_from_predictions(np.concatenate(all_p), np.concatenate(all_g), scalers)
    return per_zone, pooled


def train_benchmark_models(normalized_data, scalers, rounds, local_epochs, cent_epochs,
                           fedprox_mu, pfedme_lambda, device="cpu"):
    """
    Trains every compared model for one seed. Centralized and isolated models get cent_epochs
    passes over their data; federated clients get rounds x local_epochs passes.
    Returns {model name: {zone id: model}}, comm stats (MB), histories and loaders.
    """
    print("\n" + "=" * 85, flush=True)
    print("PHASE 2: BENCHMARK TRAINING", flush=True)
    print(f"Settings: Rounds={rounds}, Local Epochs={local_epochs}, Cent Epochs={cent_epochs}, "
          f"FedProx mu={fedprox_mu}, pFedMe lambda={pfedme_lambda}, Device={device}")
    print("=" * 85, flush=True)

    client_loaders, cent_loaders = get_dataloaders(normalized_data)
    cids = sorted(client_loaders.keys())
    model_kb = compute_model_size_kb(Seq2SeqLSTM())

    print("\n--- Centralized (pooled) ---", flush=True)
    cent_hist, cent_model = CentralizedTrainer(Seq2SeqLSTM, cent_loaders, scalers, device=device).train(epochs=cent_epochs)

    zh_hist, zh_model = CentralizedZoneHeadTrainer(client_loaders, scalers, device=device).train(epochs=cent_epochs)

    print("\n--- Isolated silos ---", flush=True)
    iso_models, _ = IsolatedTrainer(Seq2SeqLSTM, client_loaders, scalers, device=device).train_all(epochs=cent_epochs)

    def fl_clients(cls, **kw):
        return [cls(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device, **kw)
                for cid, l in client_loaders.items()]

    print("\n--- FedAvg ---", flush=True)
    server_fedavg = FederatedServer(Seq2SeqLSTM, fl_clients(FederatedClient), cent_loaders['test'], scalers, device=device)
    fedavg_hist = server_fedavg.run_federated_training(num_rounds=rounds, local_epochs=local_epochs, fedprox_mu=0.0)

    print(f"\n--- FedProx (mu={fedprox_mu}) ---", flush=True)
    server_fedprox = FederatedServer(Seq2SeqLSTM, fl_clients(FederatedClient), cent_loaders['test'], scalers, device=device)
    fedprox_hist = server_fedprox.run_federated_training(num_rounds=rounds, local_epochs=local_epochs, fedprox_mu=fedprox_mu)

    print("\n--- FedPer ---", flush=True)
    server_fedper = FedPerServer(Seq2SeqLSTM, fl_clients(FedPerClient), cent_loaders['test'], scalers, device=device)
    fedper_hist = server_fedper.run_fedper_training(num_rounds=rounds, local_epochs=local_epochs)

    print(f"\n--- pFedMe (lambda={pfedme_lambda}) ---", flush=True)
    server_pfedme = PFedMeServer(Seq2SeqLSTM, fl_clients(PFedMeClient, lamda=pfedme_lambda, K_inner=3, lr_inner=0.01),
                                 cent_loaders['test'], scalers, device=device)
    pfedme_hist = server_pfedme.run_pfedme_training(num_rounds=rounds, local_epochs=local_epochs, lamda=pfedme_lambda,
                                                    K_inner=3, lr_inner=0.01, beta=0.2)

    cv = KinematicExtrapolation(scalers, 'cv').to(device)
    ca = KinematicExtrapolation(scalers, 'ca').to(device)

    fedprox_name = f"FedProx (mu={fedprox_mu:g})"
    pfedme_name = f"pFedMe (lambda={pfedme_lambda:g})"
    zone_models = {
        'Constant velocity': {c: cv for c in cids},
        'Constant acceleration': {c: ca for c in cids},
        'Centralized (Pooled)': {c: cent_model for c in cids},
        'Centralized (zone heads)': {c: zh_model.for_zone(c) for c in cids},
        'FedAvg (Global FL)': {c: server_fedavg.global_model for c in cids},
        fedprox_name: {c: server_fedprox.global_model for c in cids},
        'FedPer (Personalized Heads)': {c: server_fedper.clients[c].model for c in cids},
        pfedme_name: {c: server_pfedme.clients[c].personalized_model for c in cids},
    }
    for k in cids:
        zone_models[SILO_NAMES[k]] = {c: iso_models[k] for c in cids}
    zone_models[OWN_ZONE] = {c: iso_models[c] for c in cids}

    # Communication in MB (2^20 bytes), uplink + downlink over all rounds and clients.
    # Centralizing = one-off upload of the 4 float32 features of every raw row used.
    raw_rows = sum(int(normalized_data[c].get('raw_rows', 0)) for c in cids)
    raw_upload_mb = round(raw_rows * 4 * 4 / 2 ** 20, 2)
    fl_mb = round(rounds * 2 * len(cids) * model_kb / 1024.0, 2)
    comm_stats = {name: 0.0 for name in zone_models}
    comm_stats.update({
        'Centralized (Pooled)': raw_upload_mb,
        'Centralized (zone heads)': raw_upload_mb,
        'FedAvg (Global FL)': fl_mb,
        fedprox_name: fl_mb,
        pfedme_name: fl_mb,
        'FedPer (Personalized Heads)': round(rounds * 2 * len(cids) * server_fedper.shared_model_size_kb / 1024.0, 2),
    })

    histories = {'cent': cent_hist, 'zone_heads': zh_hist, 'fedavg': fedavg_hist, 'fedprox': fedprox_hist,
                 'fedper': fedper_hist, 'pfedme': pfedme_hist}
    return zone_models, comm_stats, histories, (client_loaders, cent_loaders)


def metric_record(model_name, domain, m, comm, seed):
    h = m['horizons_m']
    rec = {
        'Seed': seed,
        'Model': model_name,
        'Evaluation Domain': domain,
        'ADE (m)': round(m['ADE_m'], 4),
        'FDE (m)': round(m['FDE_m'], 4),
        'RMSE (m)': round(m['RMSE_m'], 4),
        'MAE (m)': round(m['MAE_m'], 4),
        'MAPE (%)': round(m['MAPE_m'], 2),
        'ADE (ft)': round(m['ADE_ft'], 4),
        'FDE (ft)': round(m['FDE_ft'], 4),
        'Scale-normalized ADE': round(m['SNADE'], 4),
    }
    for step in range(2, PRED_STEPS + 1, 2):
        sec = step * 0.5
        rec[f'Error @ {sec:.1f}s (m)'] = round(h.get(f'FDE_{sec:.1f}s', np.nan), 4)
        rec[f'Cumulative ADE to {sec:.1f}s (m)'] = round(h.get(f'ADE_{sec:.1f}s', np.nan), 4)
    rec['Cumulative Comm (MB)'] = comm
    return rec


def evaluate_benchmark(zone_models, comm_stats, client_loaders, scalers, seed, device="cpu"):
    """
    Evaluates every model in every zone and pooled. Isolated silos are scored in all zones
    (in and out of zone); 'Isolated (own zone)' serves each zone with that zone's silo.
    Returns metric records, fairness records and per-window ADE arrays for paired tests.
    """
    print("\n" + "=" * 85, flush=True)
    print(f"PHASE 3: MULTI-DOMAIN EVALUATION (seed {seed})", flush=True)
    print("=" * 85, flush=True)

    records, fairness_records, per_window = [], [], {}
    for name, models in zone_models.items():
        per_zone, pooled = evaluate_zone_models(models, client_loaders, scalers, device)
        comm = comm_stats.get(name, 0.0)
        for cid, m in per_zone.items():
            records.append(metric_record(name, ZONES[cid], m, comm, seed))
            per_window[(name, ZONES[cid])] = m['per_window_ade_m']
        records.append(metric_record(name, POOLED, pooled, comm, seed))
        per_window[(name, POOLED)] = pooled['per_window_ade_m']

        fair = compute_fairness_metrics({ZONES[c]: m['ADE_m'] for c, m in per_zone.items()})
        fairness_records.append({
            'Seed': seed,
            'Model': name,
            'Global ADE (m)': round(pooled['ADE_m'], 4),
            'Mean Zone ADE (m)': round(fair['mean_error'], 4),
            'Worst-Zone ADE (m)': round(fair['worst_zone'][1], 4),
            'Worst Zone Name': fair['worst_zone'][0],
            'Best-Zone ADE (m)': round(fair['best_zone'][1], 4),
            'Zone Disparity Gap (m)': round(fair['gap'], 4),
            'Fairness CoV (%)': round(fair['coeff_variation'] * 100.0, 2),
            'Communication Overhead (MB)': comm
        })
        print(f"  {name:32s} pooled ADE {pooled['ADE_m']:.3f} m | " +
              " | ".join(f"{ZONES[c].split()[0]} {m['ADE_m']:.3f}" for c, m in per_zone.items()), flush=True)
    return records, fairness_records, per_window


# ==============================================================================
# 3. HYPERPARAMETER TUNING (validation split)
# ==============================================================================

def tune_hyperparameters(normalized_data, scalers, rounds, local_epochs, seed, device="cpu",
                         mu_grid=(0.001, 0.01, 0.1, 1.0), lambda_grid=(0.1, 1.0, 15.0)):
    print("\n" + "=" * 85, flush=True)
    print("TUNING FedProx mu AND pFedMe lambda ON THE VALIDATION SPLIT", flush=True)
    print("=" * 85, flush=True)
    client_loaders, cent_loaders = get_dataloaders(normalized_data)
    rows = []
    for mu in mu_grid:
        set_seed(seed)
        clients = [FederatedClient(c, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device)
                   for c, l in client_loaders.items()]
        server = FederatedServer(Seq2SeqLSTM, clients, cent_loaders['test'], scalers, device=device)
        server.run_federated_training(num_rounds=rounds, local_epochs=local_epochs, fedprox_mu=mu)
        val_ade = evaluate_model_on_loader(server.global_model, cent_loaders['val'], scalers, device)['ADE_m']
        rows.append({'Method': 'FedProx', 'Parameter': 'mu', 'Value': mu, 'Validation pooled ADE (m)': round(val_ade, 4)})
    for lam in lambda_grid:
        set_seed(seed)
        clients = [PFedMeClient(c, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device,
                                lamda=lam, K_inner=3, lr_inner=0.01) for c, l in client_loaders.items()]
        server = PFedMeServer(Seq2SeqLSTM, clients, cent_loaders['test'], scalers, device=device)
        server.run_pfedme_training(num_rounds=rounds, local_epochs=local_epochs, lamda=lam, K_inner=3, lr_inner=0.01, beta=0.2)
        _, pooled = evaluate_zone_models({c.client_id: c.personalized_model for c in server.clients},
                                         {c: {'test': l['val']} for c, l in client_loaders.items()}, scalers, device)
        rows.append({'Method': 'pFedMe', 'Parameter': 'lambda', 'Value': lam,
                     'Validation pooled ADE (m)': round(pooled['ADE_m'], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "journal_tuning_results.csv"), index=False)
    best_mu = float(df[df.Method == 'FedProx'].sort_values('Validation pooled ADE (m)').iloc[0]['Value'])
    best_lam = float(df[df.Method == 'pFedMe'].sort_values('Validation pooled ADE (m)').iloc[0]['Value'])
    print(f"[Tuning] Selected FedProx mu={best_mu}, pFedMe lambda={best_lam}", flush=True)
    return best_mu, best_lam


# ==============================================================================
# 4. STATISTICS ACROSS SEEDS
# ==============================================================================

def summarize_over_seeds(records, keys):
    df = pd.DataFrame(records)
    metrics = [c for c in df.columns if c not in keys + ['Seed', 'Worst Zone Name'] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, sort=False)[metrics]
    out = g.mean().round(4).add_suffix(' mean').join(g.std(ddof=1).round(4).add_suffix(' SD'))
    out.insert(0, 'Seeds', df.groupby(keys, sort=False).size())
    return out.reset_index()


def paired_tests(records, per_window_by_seed, comparisons):
    """
    For each (A, B, domain): paired t-test and Wilcoxon signed-rank test of ADE across seeds,
    and a Wilcoxon signed-rank test over test windows of the per-window ADE averaged over seeds
    (the split is fixed, so windows are identical across seeds).
    """
    df = pd.DataFrame(records)
    rows = []
    for a, b in comparisons:
        for domain in list(ZONES.values()) + [POOLED]:
            sa = df[(df.Model == a) & (df['Evaluation Domain'] == domain)].set_index('Seed')['ADE (m)']
            sb = df[(df.Model == b) & (df['Evaluation Domain'] == domain)].set_index('Seed')['ADE (m)']
            seeds = sorted(set(sa.index) & set(sb.index))
            if not seeds:
                continue
            xa, xb = sa.loc[seeds].values, sb.loc[seeds].values
            row = {'Model A': a, 'Model B': b, 'Domain': domain, 'Seeds': len(seeds),
                   'Mean ADE A (m)': round(xa.mean(), 4), 'Mean ADE B (m)': round(xb.mean(), 4),
                   'Mean difference A-B (m)': round((xa - xb).mean(), 4)}
            if len(seeds) >= 2 and np.any(xa != xb):
                row['Paired t p (seeds)'] = float(stats.ttest_rel(xa, xb).pvalue)
                row['Wilcoxon p (seeds)'] = float(stats.wilcoxon(xa, xb).pvalue)
            wa = np.mean([per_window_by_seed[s][(a, domain)] for s in seeds], axis=0)
            wb = np.mean([per_window_by_seed[s][(b, domain)] for s in seeds], axis=0)
            if np.any(wa != wb):
                row['Windows'] = len(wa)
                row['Wilcoxon p (windows)'] = float(stats.wilcoxon(wa, wb).pvalue)
            rows.append(row)
    return rows


# ==============================================================================
# 5. CONTROLLED HETEROGENEITY PERSONALIZATION STUDY
# ==============================================================================

def run_controlled_heterogeneity_study(normalized_data, scalers, rounds=4, local_epochs=1, device="cpu",
                                       pfedme_lambda=1.0, seed=RANDOM_SEED):
    """
    Evaluates FedAvg, FedPer and pFedMe on resampled training partitions ('low', 'medium',
    'high'); test sets are unchanged. Reports the unweighted mean of the zone ADEs.
    """
    print("\n" + "=" * 85, flush=True)
    print("PHASE 4: CONTROLLED HETEROGENEITY ANALYSIS ('low', 'medium', 'high')", flush=True)
    print("=" * 85, flush=True)

    controlled_results = []
    for level in ['low', 'medium', 'high']:
        print(f"\n--- Heterogeneity Level: {level.upper()} ---", flush=True)
        part_data = create_controlled_heterogeneity_partition(normalized_data, level=level, seed=RANDOM_SEED)
        c_loaders, cent_loaders = get_dataloaders(part_data)

        set_seed(seed)
        clients_avg = [FederatedClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device)
                       for cid, l in c_loaders.items()]
        server_avg = FederatedServer(Seq2SeqLSTM, clients_avg, cent_loaders['test'], scalers, device=device)
        server_avg.run_federated_training(num_rounds=rounds, local_epochs=local_epochs)
        m_avg = evaluate_model_on_loader(server_avg.global_model, cent_loaders['test'], scalers, device=device)
        mean_avg_ade = float(np.mean([evaluate_model_on_loader(server_avg.global_model, l['test'], scalers, device=device)['ADE_m']
                                      for l in c_loaders.values()]))

        set_seed(seed)
        clients_per = [FedPerClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device)
                       for cid, l in c_loaders.items()]
        server_per = FedPerServer(Seq2SeqLSTM, clients_per, cent_loaders['test'], scalers, device=device)
        server_per.run_fedper_training(num_rounds=rounds, local_epochs=local_epochs)
        mean_per_ade = float(np.mean([evaluate_model_on_loader(c.model, c.test_loader, scalers, device=device)['ADE_m']
                                      for c in server_per.clients]))

        set_seed(seed)
        clients_pme = [PFedMeClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device,
                                    lamda=pfedme_lambda, K_inner=3, lr_inner=0.01) for cid, l in c_loaders.items()]
        server_pme = PFedMeServer(Seq2SeqLSTM, clients_pme, cent_loaders['test'], scalers, device=device)
        server_pme.run_pfedme_training(num_rounds=rounds, local_epochs=local_epochs, lamda=pfedme_lambda,
                                       K_inner=3, lr_inner=0.01, beta=0.2)
        mean_pme_ade = float(np.mean([c.evaluate_personalized(split='test')['ADE_m'] for c in server_pme.clients]))

        controlled_results.append({
            'Heterogeneity Level': level.capitalize(),
            'Rounds': rounds,
            'FedAvg Global ADE (m)': round(m_avg['ADE_m'], 4),
            'FedAvg Mean Zone ADE (m)': round(mean_avg_ade, 4),
            'FedPer Mean Personal ADE (m)': round(mean_per_ade, 4),
            'pFedMe Mean Personal ADE (m)': round(mean_pme_ade, 4),
            'FedPer Personal Advantage (m)': round(mean_avg_ade - mean_per_ade, 4),
            'pFedMe Personal Advantage (m)': round(mean_avg_ade - mean_pme_ade, 4),
        })

    return controlled_results


# ==============================================================================
# 6. PRIVACY: ATTACKS ON TRAINED MODELS AND DP-SGD TRAINING
# ==============================================================================

def run_privacy_benchmark_study(zone_models, client_loaders, scalers, fedprox_name, pfedme_name,
                                num_attack_samples=10, attack_batch_size=1, clip_norm=1.0, device="cpu"):
    """
    DLG against the trained FL models (attacker observes one client update computed on
    attack_batch_size windows of the US-101 client), and loss-threshold MIA on every
    trained model (averaged over the three clients).
    """
    print("\n" + "=" * 85, flush=True)
    print("PHASE 5: PRIVACY ATTACKS ON TRAINED MODELS (DLG, MIA)", flush=True)
    print("=" * 85, flush=True)

    targets = [
        ('FedAvg (Global FL)', 'fedavg', zone_models['FedAvg (Global FL)'][0], [0.0, 0.5, 1.0, 2.0]),
        (fedprox_name, 'fedprox', zone_models[fedprox_name][0], [0.0]),
        (pfedme_name, 'pfedme', zone_models[pfedme_name][0], [0.0]),
        ('FedPer (Personalized Heads)', 'fedper', zone_models['FedPer (Personalized Heads)'][0], [0.0, 0.5, 1.0, 2.0]),
    ]
    dlg_records = []
    for name, method, model, sigmas in targets:
        res = run_privacy_evaluation(Seq2SeqLSTM, client_loaders, scalers, fl_method=method,
                                     num_attack_samples=num_attack_samples, dp_noise_multipliers=sigmas,
                                     device=device, model=model, attack_batch_size=attack_batch_size,
                                     clip_norm=clip_norm)
        for dp_key, m in res['reconstruction_defense'].items():
            sigma = float(dp_key.replace("dp_sigma_", ""))
            defense = ["Head gradients withheld"] if method == 'fedper' else []
            if sigma > 0:
                defense.append(f"Clipped gradient + Gaussian noise (sigma={sigma})")
            dlg_records.append({
                'FL Method': 'FedPer' if method == 'fedper' else ('FedAvg' if method == 'fedavg' else name),
                'Model': name,
                'Model State': 'trained',
                'Attack Batch Size': res['attack_batch_size'],
                'Attacked Batches': res['num_attack_batches'],
                'Defense Mechanism': " + ".join(defense) if defense else "None (Unprotected Gradients)",
                'Noise Multiplier (sigma)': sigma,
                'Position Error (std units)': round(m['position_error'], 4),
                'Input MSE': round(m['input_mse'], 4),
                'Input Cosine Similarity': round(m['input_cosine_sim'], 4),
                'Target MSE': round(m['target_mse'], 4)
            })

    mia_records = []
    for name, models in zone_models.items():
        if name.startswith('Constant'):
            continue  # no trained parameters
        aucs, accs = [], []
        for cid in sorted(client_loaders.keys()):
            r = run_membership_inference(models[cid], {cid: client_loaders[cid]}, device=device)[cid]
            aucs.append(r['auc'])
            accs.append(r['accuracy'])
        mia_records.append({'Model': name, 'MIA AUC (mean over clients)': round(float(np.mean(aucs)), 4),
                            'Attack Accuracy (%)': round(float(np.mean(accs)) * 100.0, 2)})
        print(f"  MIA {name:32s} AUC {np.mean(aucs):.4f}", flush=True)
    return dlg_records, mia_records


def run_dp_fedavg(sigmas, clip_norm, delta, normalized_data, scalers, rounds, local_epochs, seed, device="cpu"):
    """Trains DP-FedAvg (DP-SGD on every client) for each sigma and accounts epsilon with RDP."""
    print("\n" + "=" * 85, flush=True)
    print("PHASE 6: DP-FedAvg (DP-SGD + RDP accountant)", flush=True)
    print("=" * 85, flush=True)
    client_loaders, cent_loaders = get_dataloaders(normalized_data)
    model_kb = compute_model_size_kb(Seq2SeqLSTM())
    results = {}
    for sigma in sigmas:
        set_seed(seed)
        clients = [DPFederatedClient(c, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=device,
                                     noise_multiplier=sigma, clip_norm=clip_norm, seed=seed)
                   for c, l in client_loaders.items()]
        server = FederatedServer(Seq2SeqLSTM, clients, cent_loaders['test'], scalers, device=device)
        server.run_federated_training(num_rounds=rounds, local_epochs=local_epochs)
        accounting = [{'Client': c.name, 'q': round(c.sample_rate, 5), 'Steps': c.steps,
                       'Epsilon': round(c.epsilon(delta), 3)} for c in clients]
        eps = max(a['Epsilon'] for a in accounting)
        name = f"DP-FedAvg (sigma={sigma:g}, C={clip_norm:g})"
        results[name] = {'models': {c: server.global_model for c in client_loaders},
                         'epsilon': eps, 'delta': delta, 'accounting': accounting,
                         'comm': round(rounds * 2 * len(clients) * model_kb / 1024.0, 2)}
        print(f"[DP] {name}: epsilon = {eps:.3f} at delta = {delta} (max over clients)", flush=True)
    return results


# ==============================================================================
# 7. INTEGRATED TRADE-OFF TABLE (measured values only)
# ==============================================================================

def generate_tradeoff_summary(fairness_summary, dlg_records, mia_records, dp_results):
    """
    One row per configuration. Values that were not measured are NaN (exported as N/A);
    nothing is copied between methods or extrapolated.
    """
    dlg_clean = {r['Model']: r for r in dlg_records if r['Noise Multiplier (sigma)'] == 0.0}
    mia = {r['Model']: r['MIA AUC (mean over clients)'] for r in mia_records}
    rows = []
    for _, f in fairness_summary.iterrows():
        name = f['Model']
        if name in dp_results:
            eps = f"{dp_results[name]['epsilon']:.3f} (delta={dp_results[name]['delta']:g})"
        elif name.startswith(('FedAvg', 'FedProx', 'FedPer', 'pFedMe')):
            eps = 'None (no DP)'
        else:
            eps = 'N/A (no gradient sharing)'
        d = dlg_clean.get(name, {})
        rows.append({
            'Model Configuration': name,
            'Global ADE (m)': f['Global ADE (m) mean'],
            'Worst-Zone ADE (m)': f['Worst-Zone ADE (m) mean'],
            'Fairness CoV (%)': f['Fairness CoV (%) mean'],
            'Comm Overhead (MB)': f['Communication Overhead (MB) mean'],
            'DLG Position Error (std units)': d.get('Position Error (std units)', np.nan),
            'DLG Cosine Similarity': d.get('Input Cosine Similarity', np.nan),
            'MIA AUC': mia.get(name, np.nan),
            'Privacy Budget (epsilon)': eps
        })
    return rows


# ==============================================================================
# 8. EXPORT AND MARKDOWN REPORT
# ==============================================================================

def df_to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("N/A" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def export_results(outputs, run_info):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = {
        'benchmark': "journal_benchmark_results.csv",
        'benchmark_summary': "journal_benchmark_summary.csv",
        'fairness': "journal_fairness_results.csv",
        'fairness_summary': "journal_fairness_summary.csv",
        'tests': "journal_statistical_tests.csv",
        'controlled': "journal_controlled_heterogeneity_results.csv",
        'dlg': "journal_privacy_dlg_results.csv",
        'mia': "journal_privacy_mia_results.csv",
        'dp': "journal_dp_accounting.csv",
        'tradeoff': "journal_tradeoff_matrix.csv",
        'pairwise': "journal_pairwise_divergence.csv",
    }
    dfs = {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)) for k, v in outputs.items()}
    for k, fname in files.items():
        if k in dfs:
            dfs[k].to_csv(os.path.join(OUTPUT_DIR, fname), index=False, na_rep='N/A')
    print(f"\n[Export] Result CSVs written to {OUTPUT_DIR}", flush=True)

    print("\n[Visualization] Generating figures...", flush=True)
    try:
        h = run_info.get('histories')
        if h is not None:
            plot_learning_curves(h['fedavg'], cent_history=h.get('cent'), fedprox_history=h.get('fedprox'),
                                 fedper_history=h.get('fedper'), pfedme_history=h.get('pfedme'))
        plot_heterogeneity_vs_personalization(dfs['controlled'])
        plot_privacy_leakage_vs_dp(dfs['dlg'][dfs['dlg']['FL Method'].isin(['FedAvg', 'FedPer'])])
        fs = dfs['fairness_summary'].rename(columns=lambda c: c[:-5] if c.endswith(' mean') else c)
        plot_fairness_disparity_comparison(fs)
        plot_accuracy_privacy_tradeoff_pareto(dfs['tradeoff'])
    except Exception as e:
        print(f"[Visualization Warning] Error generating plots: {e}", flush=True)

    pooled = dfs['benchmark_summary'][dfs['benchmark_summary']['Evaluation Domain'] == POOLED]
    keep = ['Model', 'Seeds', 'ADE (m) mean', 'ADE (m) SD', 'FDE (m) mean', 'FDE (m) SD',
            'Scale-normalized ADE mean', 'Cumulative Comm (MB) mean']
    report = f"""# Journal Benchmark Report

Generated by `run_journal_experiments.py`. All values are means over seeds {run_info['seeds']} (SD over seeds,
ddof=1). Observation {OBS_STEPS * 0.5:.1f} s, prediction {PRED_STEPS * 0.5:.1f} s at 2 Hz.
Rounds: {run_info['rounds']} x {run_info['local_epochs']} local epoch(s); centralized and isolated epochs:
{run_info['cent_epochs']} (equal budget). FedProx mu = {run_info['mu']}, pFedMe lambda = {run_info['lam']}
({'tuned on validation' if run_info['tuned'] else 'not tuned'}).

Aggregate JS divergence: {run_info['hetero']['natural_score']:.4f}; gradient cosine distance (one batch,
initial model): {run_info['hetero']['gradient_divergence']:.4f}. Per-pair KL / JS / EMD: `journal_pairwise_divergence.csv`.

## Pooled test set

{df_to_markdown(pooled[[c for c in keep if c in pooled.columns]])}

Per-zone results: `journal_benchmark_summary.csv` (all seeds: `journal_benchmark_results.csv`).

## Fairness across zones

{df_to_markdown(dfs['fairness_summary'][[c for c in dfs['fairness_summary'].columns if not c.endswith(' SD')]])}

## Paired tests

{df_to_markdown(dfs['tests']) if len(dfs['tests']) else 'Not enough seeds for paired tests.'}

## Controlled heterogeneity (seed {run_info['seeds'][0]})

{df_to_markdown(dfs['controlled'])}

## Privacy (seed {run_info['seeds'][0]})

DLG against trained models; position errors are in standardized input units. Noise rows clip the
observed gradient and add Gaussian noise once; they are not DP training and carry no epsilon.

{df_to_markdown(dfs['dlg'])}

MIA (loss threshold, members = client training windows, non-members = client test windows):

{df_to_markdown(dfs['mia'])}

DP-FedAvg accounting (DP-SGD, Poisson sampling, RDP accountant; record-level, not vehicle-level):

{df_to_markdown(dfs['dp']) if len(dfs.get('dp', [])) else 'DP training not run.'}

## Trade-off matrix (measured values only)

{df_to_markdown(dfs['tradeoff'])}
"""
    for path in [os.path.join(OUTPUT_DIR, "JOURNAL_BENCHMARK_REPORT.md"), os.path.join(BASE_DIR, "JOURNAL_BENCHMARK_REPORT.md")]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
    print(f"[Report] Markdown summary written to {OUTPUT_DIR}", flush=True)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    args = parse_args()
    t_start = time.time()

    rounds, local_epochs = args.rounds, args.local_epochs
    attack_samples, seeds, dp_sigmas = args.attack_samples, args.seeds, args.dp_sigmas
    if args.fast:
        rounds, attack_samples, seeds, dp_sigmas = 2, 2, seeds[:2], dp_sigmas[:1]
    cent_epochs = args.cent_epochs if args.cent_epochs is not None else rounds * local_epochs

    print("=" * 85, flush=True)
    print("SMART CITIES FEDERATED TRAJECTORY PREDICTION: MASTER JOURNAL PIPELINE", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {args.device} | Seeds: {seeds}", flush=True)
    print(f"Rounds: {rounds} | Local epochs: {local_epochs} | Cent epochs: {cent_epochs} | "
          f"DLG batches: {attack_samples} x {args.attack_batch_size}", flush=True)
    print("=" * 85, flush=True)

    preprocessor = NGSIMPreprocessor()
    normalized_data, scalers = preprocessor.prepare_dataset(force_recompute=args.force_preprocess)

    hetero = run_heterogeneity_study(normalized_data, scalers, device=args.device)

    mu, lam = args.fedprox_mu, args.pfedme_lambda
    if args.tune:
        mu, lam = tune_hyperparameters(normalized_data, scalers, rounds, local_epochs, seeds[0], device=args.device)
    fedprox_name, pfedme_name = f"FedProx (mu={mu:g})", f"pFedMe (lambda={lam:g})"

    all_records, all_fairness, per_window_by_seed = [], [], {}
    first = None
    for seed in seeds:
        set_seed(seed)
        zone_models, comm_stats, histories, (client_loaders, cent_loaders) = train_benchmark_models(
            normalized_data, scalers, rounds, local_epochs, cent_epochs, mu, lam, device=args.device)
        records, fairness, per_window = evaluate_benchmark(zone_models, comm_stats, client_loaders, scalers, seed,
                                                           device=args.device)
        all_records += records
        all_fairness += fairness
        per_window_by_seed[seed] = per_window
        if first is None:
            first = (zone_models, histories, client_loaders)

    zone_models, histories, client_loaders = first

    # DP-FedAvg (first seed): evaluated like the other models and added to the tables
    dp_results = {}
    if dp_sigmas:
        dp_results = run_dp_fedavg(dp_sigmas, args.dp_clip, args.dp_delta, normalized_data, scalers,
                                   rounds, local_epochs, seeds[0], device=args.device)
        for name, r in dp_results.items():
            recs, fair, pw = evaluate_benchmark({name: r['models']}, {name: r['comm']}, client_loaders, scalers,
                                                seeds[0], device=args.device)
            all_records += recs
            all_fairness += fair
            per_window_by_seed[seeds[0]].update(pw)

    comparisons = [('FedPer (Personalized Heads)', OWN_ZONE), ('FedPer (Personalized Heads)', 'FedAvg (Global FL)'),
                   ('FedPer (Personalized Heads)', 'Centralized (zone heads)'),
                   ('FedPer (Personalized Heads)', 'Centralized (Pooled)'),
                   ('FedPer (Personalized Heads)', 'Constant velocity'), (OWN_ZONE, 'Constant velocity')]
    tests = paired_tests([r for r in all_records if not r['Model'].startswith('DP-')], per_window_by_seed, comparisons)

    controlled = run_controlled_heterogeneity_study(normalized_data, scalers, rounds=rounds, local_epochs=local_epochs,
                                                    device=args.device, pfedme_lambda=lam, seed=seeds[0])

    dlg_records, mia_records = run_privacy_benchmark_study(
        zone_models, client_loaders, scalers, fedprox_name, pfedme_name, num_attack_samples=attack_samples,
        attack_batch_size=args.attack_batch_size, clip_norm=args.dp_clip, device=args.device)
    for name, r in dp_results.items():
        aucs = [run_membership_inference(r['models'][c], {c: client_loaders[c]}, device=args.device)[c]['auc']
                for c in sorted(client_loaders)]
        mia_records.append({'Model': name, 'MIA AUC (mean over clients)': round(float(np.mean(aucs)), 4),
                            'Attack Accuracy (%)': np.nan})

    benchmark_summary = summarize_over_seeds(all_records, ['Model', 'Evaluation Domain'])
    fairness_summary = summarize_over_seeds(all_fairness, ['Model'])
    tradeoff = generate_tradeoff_summary(fairness_summary, dlg_records, mia_records, dp_results)
    dp_rows = [dict(Configuration=n, **a) for n, r in dp_results.items() for a in r['accounting']]

    export_results({
        'benchmark': all_records, 'benchmark_summary': benchmark_summary,
        'fairness': all_fairness, 'fairness_summary': fairness_summary,
        'tests': tests, 'controlled': controlled, 'dlg': dlg_records, 'mia': mia_records,
        'dp': dp_rows, 'tradeoff': tradeoff, 'pairwise': hetero['pairwise_records'],
    }, {'seeds': seeds, 'rounds': rounds, 'local_epochs': local_epochs, 'cent_epochs': cent_epochs,
        'mu': mu, 'lam': lam, 'tuned': args.tune, 'hetero': hetero, 'histories': histories})

    total_time = time.time() - t_start
    print("\n" + "=" * 85, flush=True)
    print(f"JOURNAL EXPERIMENTAL PIPELINE COMPLETE IN {total_time:.2f}s ({total_time/60.0:.2f} mins)", flush=True)
    print("=" * 85, flush=True)


if __name__ == '__main__':
    main()
