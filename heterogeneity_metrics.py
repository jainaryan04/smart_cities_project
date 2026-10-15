"""
Heterogeneity Quantification Module for Geographic Non-IID Analysis.
Computes statistical divergence metrics between federated client data distributions
to formally measure and control geographic heterogeneity levels.
"""

import copy
import os
from collections import OrderedDict
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from config import CLIENT_NAMES, CACHE_DIR, RANDOM_SEED


# Feature Index Mapping
FEATURE_INDICES = {
    'displacement_x': 0,
    'displacement_y': 1,
    'speed': 2,
    'acceleration': 3,
    'local_x_rel': 0,
    'local_y_rel': 1,
    'v_vel': 2,
    'v_acc': 3
}

INDEX_TO_FEATURE_NAME = {
    0: 'displacement_x',
    1: 'displacement_y',
    2: 'speed',
    3: 'acceleration'
}


def _load_cached_scalers():
    """Attempt to load feature normalization scalers from cache."""
    cache_path = os.path.join(CACHE_DIR, "federated_trajectory_data.npz")
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            if 'x_mean' in data and 'x_std' in data:
                return {
                    'x_mean': data['x_mean'],
                    'x_std': data['x_std']
                }
        except Exception:
            pass
    return None


def _get_feature_values(client_dict, feature_idx, raw=False, scalers=None):
    """
    Extracts 1D flattened feature values for a specific client.
    
    Args:
        client_dict (dict): Dictionary for a single client with 'train_x' array.
        feature_idx (int): Index of the feature (0 to 3).
        raw (bool): If True, returns unscaled physical values if available.
        scalers (dict, optional): Scalers dictionary with 'x_mean' and 'x_std'.
        
    Returns:
        np.ndarray: 1D array of feature values.
    """
    if raw and 'raw_train_x' in client_dict:
        return np.asarray(client_dict['raw_train_x'])[..., feature_idx].flatten()
    
    norm_vals = np.asarray(client_dict['train_x'])[..., feature_idx].flatten()
    
    if raw:
        if scalers is None:
            scalers = _load_cached_scalers()
        if scalers is not None and 'x_mean' in scalers and 'x_std' in scalers:
            x_mean = scalers['x_mean'][feature_idx]
            x_std = scalers['x_std'][feature_idx]
            return norm_vals * x_std + x_mean
            
    return norm_vals


def compute_feature_distributions(client_data_dict, feature_idx, num_bins=50):
    """
    For each client, extract feature at index feature_idx from training inputs
    (shape [N, T_obs, 4]), flatten across time, and create a histogram distribution.
    Feature indices: 0=Local_X_rel, 1=Local_Y_rel, 2=v_Vel, 3=v_Acc.
    
    A common bin range is computed across all clients to ensure valid probability
    distribution alignment for divergence comparisons.
    
    Args:
        client_data_dict (dict): {cid: {'train_x': np.ndarray, ...}}
        feature_idx (int): Feature index (0 to 3).
        num_bins (int): Number of histogram bins (default: 50).
        
    Returns:
        dict: {client_id: (hist_values, bin_edges)} where hist_values sums to 1.0.
    """
    client_features = {}
    for cid, cdata in client_data_dict.items():
        vals = np.asarray(cdata['train_x'])[..., feature_idx].flatten()
        client_features[cid] = vals

    # Compute common bin range across all clients
    all_vals = np.concatenate(list(client_features.values()))
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    if np.isclose(min_val, max_val):
        max_val += 1e-5
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    distributions = {}
    for cid, vals in client_features.items():
        hist, _ = np.histogram(vals, bins=bin_edges)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            prob = hist / total
        else:
            prob = np.ones_like(hist) / len(hist)
        distributions[cid] = (prob, bin_edges)

    return distributions


def compute_kl_divergence(dist_p, dist_q, epsilon=1e-10):
    """
    Computes Kullback-Leibler divergence KL(P || Q) = sum(P * log(P / Q)).
    Adds epsilon smoothing to prevent log(0) and division by zero.
    
    Args:
        dist_p (np.ndarray): 1D probability distribution P.
        dist_q (np.ndarray): 1D probability distribution Q.
        epsilon (float): Smoothing parameter (default: 1e-10).
        
    Returns:
        float: KL divergence value.
    """
    p = np.asarray(dist_p, dtype=np.float64) + epsilon
    q = np.asarray(dist_q, dtype=np.float64) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl = np.sum(p * np.log(p / q))
    return float(max(0.0, kl))


def compute_js_divergence(dist_p, dist_q):
    """
    Computes Jensen-Shannon divergence between two distributions.
    Uses scipy.spatial.distance.jensenshannon and squares the result
    to obtain the true divergence (scipy returns the JS distance = sqrt(JS divergence)).
    
    Args:
        dist_p (np.ndarray): 1D probability distribution P.
        dist_q (np.ndarray): 1D probability distribution Q.
        
    Returns:
        float: JS divergence value in [0, 1].
    """
    p = np.asarray(dist_p, dtype=np.float64)
    q = np.asarray(dist_q, dtype=np.float64)
    
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum
        
    js_distance = jensenshannon(p, q)
    if np.isnan(js_distance):
        return 0.0
    js_div = float(js_distance ** 2)
    return float(np.clip(js_div, 0.0, 1.0))


def compute_emd(values_p, values_q):
    """
    Computes Earth Mover's Distance (Wasserstein-1 distance) between raw feature value arrays.
    Uses scipy.stats.wasserstein_distance.
    
    Args:
        values_p (np.ndarray): 1D array of sample values from distribution P.
        values_q (np.ndarray): 1D array of sample values from distribution Q.
        
    Returns:
        float: Wasserstein-1 distance.
    """
    vp = np.asarray(values_p, dtype=np.float64).flatten()
    vq = np.asarray(values_q, dtype=np.float64).flatten()
    return float(wasserstein_distance(vp, vq))


def compute_pairwise_heterogeneity(client_data_dict, features_to_analyze=None, scalers=None):
    """
    Computes ALL divergence metrics (KL, JS, EMD) for ALL pairs of clients across
    specified features (default: speed=2, acceleration=3, displacement_y=1).
    
    Uses normalized data for histogram distributions (KL, JS) and raw feature
    values for EMD.
    
    Args:
        client_data_dict (dict): {cid: {'train_x': np.ndarray, ...}}
        features_to_analyze (dict or list, optional): Mapping of feature name to index,
            or list of feature names/indices. Defaults to speed, acceleration, displacement_y.
        scalers (dict, optional): Normalization scalers for unscaling to raw units.
        
    Returns:
        dict: Nested dictionary of format:
            {(client_i, client_j): {'feature_name': {'kl': val, 'js': val, 'emd': val}}}
    """
    if features_to_analyze is None:
        features_dict = OrderedDict([
            ('speed', 2),
            ('acceleration', 3),
            ('displacement_y', 1)
        ])
    elif isinstance(features_to_analyze, dict):
        features_dict = OrderedDict(features_to_analyze)
    elif isinstance(features_to_analyze, (list, tuple)):
        features_dict = OrderedDict()
        for item in features_to_analyze:
            if isinstance(item, int):
                name = INDEX_TO_FEATURE_NAME.get(item, f"feature_{item}")
                features_dict[name] = item
            elif isinstance(item, str):
                idx = FEATURE_INDICES.get(item.lower(), None)
                if idx is None:
                    raise ValueError(f"Unknown feature name: {item}")
                features_dict[item] = idx
    else:
        raise ValueError("features_to_analyze must be None, a dict, or a list.")

    if scalers is None:
        scalers = _load_cached_scalers()

    client_ids = sorted(list(client_data_dict.keys()))
    
    # Precompute distributions and raw values per feature
    feature_dists = {}
    feature_raw_vals = {}
    for feat_name, feat_idx in features_dict.items():
        feature_dists[feat_name] = compute_feature_distributions(client_data_dict, feat_idx)
        feature_raw_vals[feat_name] = {
            cid: _get_feature_values(client_data_dict[cid], feat_idx, raw=True, scalers=scalers)
            for cid in client_ids
        }

    pairwise_metrics = {}
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            cid_i = client_ids[i]
            cid_j = client_ids[j]
            pair_key = (cid_i, cid_j)
            pairwise_metrics[pair_key] = {}

            for feat_name in features_dict:
                dist_i, _ = feature_dists[feat_name][cid_i]
                dist_j, _ = feature_dists[feat_name][cid_j]
                
                raw_i = feature_raw_vals[feat_name][cid_i]
                raw_j = feature_raw_vals[feat_name][cid_j]

                kl_val = compute_kl_divergence(dist_i, dist_j)
                js_val = compute_js_divergence(dist_i, dist_j)
                emd_val = compute_emd(raw_i, raw_j)

                pairwise_metrics[pair_key][feat_name] = {
                    'kl': kl_val,
                    'js': js_val,
                    'emd': emd_val
                }

    return pairwise_metrics


def compute_overall_heterogeneity_score(pairwise_metrics):
    """
    Computes an aggregate heterogeneity score by averaging JS divergence
    across all pairs and features.
    
    Args:
        pairwise_metrics (dict): Output from compute_pairwise_heterogeneity.
        
    Returns:
        float: Aggregate JS divergence heterogeneity score.
    """
    js_values = []
    for pair, feats in pairwise_metrics.items():
        for feat_name, metric_dict in feats.items():
            if 'js' in metric_dict:
                js_values.append(metric_dict['js'])
                
    if not js_values:
        return 0.0
    return float(np.mean(js_values))


def compute_gradient_divergence(client_gradients_dict):
    """
    Computes pairwise gradient divergence across clients.
    Takes dict {client_id: gradient_dict (state_dict-like)}, computes pairwise
    cosine similarity of flattened gradient vectors, and returns the mean
    pairwise cosine distance (1 - cosine_similarity).
    
    Args:
        client_gradients_dict (dict): {cid: {layer_name: tensor_or_array, ...}}
        
    Returns:
        float: Mean pairwise cosine distance in [0, 2].
    """
    flat_grads = {}
    for cid, g_dict in client_gradients_dict.items():
        tensor_list = []
        for key in sorted(g_dict.keys()):
            val = g_dict[key]
            if isinstance(val, torch.Tensor):
                t = val.detach().cpu().flatten().float()
            elif isinstance(val, np.ndarray):
                t = torch.from_numpy(val).flatten().float()
            else:
                continue
            tensor_list.append(t)
            
        if tensor_list:
            flat_grads[cid] = torch.cat(tensor_list)
        else:
            flat_grads[cid] = torch.tensor([])

    client_ids = sorted(list(flat_grads.keys()))
    if len(client_ids) < 2:
        return 0.0

    cosine_distances = []
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            vec_i = flat_grads[client_ids[i]]
            vec_j = flat_grads[client_ids[j]]
            
            if len(vec_i) == 0 or len(vec_j) == 0:
                continue
                
            norm_i = torch.norm(vec_i)
            norm_j = torch.norm(vec_j)
            
            if norm_i < 1e-12 or norm_j < 1e-12:
                cos_sim = 1.0  # Zero updates: orthogonal or identical
            else:
                cos_sim = torch.dot(vec_i, vec_j) / (norm_i * norm_j)
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0).item()
                
            cosine_distances.append(1.0 - cos_sim)

    if not cosine_distances:
        return 0.0
    return float(np.mean(cosine_distances))


def create_controlled_heterogeneity_partition(normalized_data, level='medium', seed=RANDOM_SEED):
    """
    Creates controlled heterogeneity scenarios across federated clients:
    
    - 'low': Mix 30% of trajectories between zones to create overlap.
             Randomly sample 30% from each zone and redistribute uniformly.
    - 'medium': Return data as-is (natural geographic partition).
    - 'high': For each zone, remove trajectories whose speed/displacement
              profiles overlap with other zones. Keep only the most
              zone-distinctive trajectories (those in the tails of inter-zone distance).
              
    Args:
        normalized_data (dict): Dict of client datasets with keys 'train_x', 'train_y', etc.
        level (str): 'low', 'medium', or 'high'.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: Modified copy of normalized_data with controlled heterogeneity.
    """
    modified_data = copy.deepcopy(normalized_data)
    client_ids = sorted(list(modified_data.keys()))
    rng = np.random.default_rng(seed)

    if level == 'medium':
        return modified_data

    elif level == 'low':
        # Mix 30% of trajectories between zones
        mix_fraction = 0.30
        pooled_data = {'train_x': [], 'train_y': []}
        has_raw_y = 'raw_train_y' in modified_data[client_ids[0]]
        if has_raw_y:
            pooled_data['raw_train_y'] = []

        donor_counts = {}
        for cid in client_ids:
            n_samples = len(modified_data[cid]['train_x'])
            k_mix = int(np.round(mix_fraction * n_samples))
            donor_counts[cid] = k_mix

            perm = rng.permutation(n_samples)
            keep_idx = perm[:-k_mix] if k_mix > 0 else perm
            donor_idx = perm[-k_mix:] if k_mix > 0 else np.array([], dtype=int)

            # Store pooled donor samples
            if k_mix > 0:
                pooled_data['train_x'].append(modified_data[cid]['train_x'][donor_idx])
                pooled_data['train_y'].append(modified_data[cid]['train_y'][donor_idx])
                if has_raw_y:
                    pooled_data['raw_train_y'].append(modified_data[cid]['raw_train_y'][donor_idx])

            # Retain non-mixed 70% samples for now
            modified_data[cid]['train_x'] = modified_data[cid]['train_x'][keep_idx]
            modified_data[cid]['train_y'] = modified_data[cid]['train_y'][keep_idx]
            if has_raw_y:
                modified_data[cid]['raw_train_y'] = modified_data[cid]['raw_train_y'][keep_idx]

        if any(c > 0 for c in donor_counts.values()):
            # Concatenate and shuffle pool
            pool_x = np.concatenate(pooled_data['train_x'], axis=0)
            pool_y = np.concatenate(pooled_data['train_y'], axis=0)
            pool_perm = rng.permutation(len(pool_x))
            pool_x = pool_x[pool_perm]
            pool_y = pool_y[pool_perm]
            if has_raw_y:
                pool_raw_y = np.concatenate(pooled_data['raw_train_y'], axis=0)[pool_perm]

            # Redistribute back to clients
            offset = 0
            for cid in client_ids:
                k_mix = donor_counts[cid]
                if k_mix > 0:
                    recv_x = pool_x[offset:offset + k_mix]
                    recv_y = pool_y[offset:offset + k_mix]
                    modified_data[cid]['train_x'] = np.concatenate([modified_data[cid]['train_x'], recv_x], axis=0)
                    modified_data[cid]['train_y'] = np.concatenate([modified_data[cid]['train_y'], recv_y], axis=0)
                    if has_raw_y:
                        recv_raw_y = pool_raw_y[offset:offset + k_mix]
                        modified_data[cid]['raw_train_y'] = np.concatenate([modified_data[cid]['raw_train_y'], recv_raw_y], axis=0)
                    offset += k_mix

                # Final shuffle within client
                n_total = len(modified_data[cid]['train_x'])
                shuf = rng.permutation(n_total)
                modified_data[cid]['train_x'] = modified_data[cid]['train_x'][shuf]
                modified_data[cid]['train_y'] = modified_data[cid]['train_y'][shuf]
                if has_raw_y:
                    modified_data[cid]['raw_train_y'] = modified_data[cid]['raw_train_y'][shuf]

        return modified_data

    elif level == 'high':
        # Remove trajectories that overlap with other zones
        # Compute trajectory-level kinematic profiles: [mean_speed, mean_acc, displacement_y]
        client_profiles = {}
        for cid in client_ids:
            tx = modified_data[cid]['train_x']  # [N, T_obs, 4]
            mean_speed = np.mean(tx[..., 2], axis=1)        # [N]
            mean_acc = np.mean(tx[..., 3], axis=1)          # [N]
            mean_disp_y = np.mean(tx[..., 1], axis=1)       # [N]
            client_profiles[cid] = np.stack([mean_speed, mean_acc, mean_disp_y], axis=1)  # [N, 3]

        # Compute centroid for each zone in profile space
        zone_centroids = {cid: np.mean(client_profiles[cid], axis=0) for cid in client_ids}

        # For each client, retain the most distinctive trajectories (farthest from other zone centroids)
        keep_ratio = 0.70  # Prune 30% of the most overlapping trajectories
        for cid in client_ids:
            profiles = client_profiles[cid]  # [N, 3]
            other_cids = [c for c in client_ids if c != cid]
            if not other_cids:
                continue

            # Compute minimum Euclidean distance to any other zone's centroid
            other_centers = np.array([zone_centroids[c] for c in other_cids])  # [num_other, 3]
            # dist shape: [N, num_other]
            dists = np.linalg.norm(profiles[:, np.newaxis, :] - other_centers[np.newaxis, :, :], axis=2)
            min_dist_to_others = np.min(dists, axis=1)  # [N]

            # Highest distance = least overlap with other zones (tails of inter-zone distance)
            n_keep = max(10, int(np.round(keep_ratio * len(profiles))))
            distinctive_idx = np.argsort(min_dist_to_others)[-n_keep:]

            # Apply pruning
            modified_data[cid]['train_x'] = modified_data[cid]['train_x'][distinctive_idx]
            modified_data[cid]['train_y'] = modified_data[cid]['train_y'][distinctive_idx]
            if 'raw_train_y' in modified_data[cid]:
                modified_data[cid]['raw_train_y'] = modified_data[cid]['raw_train_y'][distinctive_idx]

        return modified_data

    else:
        raise ValueError(f"Unknown heterogeneity level '{level}'. Must be 'low', 'medium', or 'high'.")


def summarize_zone_statistics(client_data_dict, scalers=None):
    """
    For each client, computes summary statistics (mean, std, median) of:
    - speed (v_Vel)
    - acceleration (v_Acc)
    - displacement_x (Local_X_rel)
    - displacement_y (Local_Y_rel)
    - trajectory_length (computed as cumulative step distance across observation steps)
    
    Args:
        client_data_dict (dict): {cid: {'train_x': np.ndarray, ...}}
        scalers (dict, optional): Normalization scalers to compute unscaled physical values.
        
    Returns:
        dict: DataFrame-like dictionary indexed by client ID.
    """
    if scalers is None:
        scalers = _load_cached_scalers()

    zone_stats = OrderedDict()
    
    for cid in sorted(client_data_dict.keys()):
        cdict = client_data_dict[cid]
        cname = cdict.get('name', CLIENT_NAMES.get(cid, f"Client_{cid}"))
        
        # Extract features (unscaled if scalers available)
        speed = _get_feature_values(cdict, 2, raw=True, scalers=scalers)
        acc = _get_feature_values(cdict, 3, raw=True, scalers=scalers)
        disp_x = _get_feature_values(cdict, 0, raw=True, scalers=scalers)
        disp_y = _get_feature_values(cdict, 1, raw=True, scalers=scalers)

        # Compute trajectory length across the observation horizon
        # Use unscaled coordinate steps if available
        tx = np.asarray(cdict.get('raw_train_x', cdict['train_x']))
        if 'raw_train_x' not in cdict and scalers is not None:
            # tx shape [N, T_obs, 4]
            unscaled_x = tx[..., 0] * scalers['x_std'][0] + scalers['x_mean'][0]
            unscaled_y = tx[..., 1] * scalers['x_std'][1] + scalers['x_mean'][1]
        else:
            unscaled_x = tx[..., 0]
            unscaled_y = tx[..., 1]

        dx = np.diff(unscaled_x, axis=1)
        dy = np.diff(unscaled_y, axis=1)
        step_dists = np.sqrt(dx ** 2 + dy ** 2)
        traj_lengths = np.sum(step_dists, axis=1)  # [N]

        zone_stats[cid] = {
            'client_id': cid,
            'client_name': cname,
            'num_trajectories': len(tx),
            'speed_mean': float(np.mean(speed)),
            'speed_std': float(np.std(speed)),
            'speed_median': float(np.median(speed)),
            'acc_mean': float(np.mean(acc)),
            'acc_std': float(np.std(acc)),
            'acc_median': float(np.median(acc)),
            'displacement_x_mean': float(np.mean(disp_x)),
            'displacement_x_std': float(np.std(disp_x)),
            'displacement_x_median': float(np.median(disp_x)),
            'displacement_y_mean': float(np.mean(disp_y)),
            'displacement_y_std': float(np.std(disp_y)),
            'displacement_y_median': float(np.median(disp_y)),
            'trajectory_length_mean': float(np.mean(traj_lengths)),
            'trajectory_length_std': float(np.std(traj_lengths)),
            'trajectory_length_median': float(np.median(traj_lengths)),
        }

    return zone_stats


# Alias for backward- and cross-compatibility
compute_zone_kinematic_stats = summarize_zone_statistics


def print_heterogeneity_report(pairwise_metrics, zone_stats, overall_score):
    """
    Prints a formatted report of all heterogeneity metrics including:
    - Overall Heterogeneity Score (Aggregate JS Divergence)
    - Zone Kinematic Statistics Summary
    - Pairwise Statistical Divergences (KL, JS, EMD)
    
    Args:
        pairwise_metrics (dict): Pairwise divergence dictionary.
        zone_stats (dict): Per-client zone statistics.
        overall_score (float): Aggregate heterogeneity score.
    """
    print("\n" + "=" * 90)
    print(" " * 22 + "GEOGRAPHIC NON-IID HETEROGENEITY REPORT")
    print("=" * 90)
    print(f"Overall Heterogeneity Score (Aggregate JS Divergence): {overall_score:.4f}")
    if overall_score < 0.15:
        het_label = "LOW (Homogeneous / High Inter-zone Overlap)"
    elif overall_score < 0.35:
        het_label = "MEDIUM (Moderate Domain Shift / Distinct Road Geometries)"
    else:
        het_label = "HIGH (Severe Non-IID / Extreme Kinematic Divergence)"
    print(f"Heterogeneity Classification: {het_label}")
    print("-" * 90)

    # 1. Zone Statistics
    print("\n[1] Zone-Level Kinematic & Geometric Profiles:")
    header = f"{'Zone / Client':<28} | {'Speed (ft/s)':<17} | {'Acc (ft/s^2)':<17} | {'Traj Length (ft)':<17}"
    print(header)
    print("-" * len(header))
    for cid, stats_data in zone_stats.items():
        name = stats_data.get('client_name', f"Client {cid}")
        s_str = f"{stats_data['speed_mean']:.2f} +/- {stats_data['speed_std']:.2f}"
        a_str = f"{stats_data['acc_mean']:.2f} +/- {stats_data['acc_std']:.2f}"
        t_str = f"{stats_data['trajectory_length_mean']:.2f} +/- {stats_data['trajectory_length_std']:.2f}"
        print(f"{name:<28} | {s_str:<17} | {a_str:<17} | {t_str:<17}")

    # 2. Pairwise Divergence Metrics
    print("\n[2] Pairwise Statistical Divergence Across Geographic Zones:")
    pw_header = f"{'Client Pair':<24} | {'Feature':<16} | {'KL Div':<10} | {'JS Div':<10} | {'EMD (ft)':<12}"
    print(pw_header)
    print("-" * len(pw_header))
    for (cid_i, cid_j), feats in pairwise_metrics.items():
        pair_name = f"Client {cid_i} vs {cid_j}"
        for f_name, metrics in feats.items():
            kl = metrics.get('kl', 0.0)
            js = metrics.get('js', 0.0)
            emd = metrics.get('emd', 0.0)
            print(f"{pair_name:<24} | {f_name:<16} | {kl:<10.4f} | {js:<10.4f} | {emd:<12.4f}")
        print("-" * len(pw_header))

    print("=" * 90 + "\n")


if __name__ == '__main__':
    print("Heterogeneity metrics module loaded successfully")
