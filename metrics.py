"""
Evaluation Metrics for Vehicle Trajectory Prediction.
Metrics implemented:
- Average Displacement Error (ADE) in meters & feet
- Final Displacement Error (FDE) in meters & feet
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Zone Fairness Metrics
- Personalization Recovery Rate
- Horizon-specific ADE breakdown (e.g., at 1.0s, 2.0s, 3.0s)
- Unscaling utilities from normalized space to physical real-world coordinates
"""

import numpy as np
import torch
from config import FEET_TO_METERS, OBS_STEPS, PRED_STEPS


def unscale_coordinates(norm_coords, y_mean, y_std):
    """
    Converts normalized coordinates back to real physical coordinates (feet).
    norm_coords: [N, T, 2] or tensor
    y_mean: [2]
    y_std: [2]
    """
    if isinstance(norm_coords, torch.Tensor):
        norm_coords = norm_coords.detach().cpu().numpy()
    
    # y_mean and y_std are for the 2 target coordinate dimensions
    unscaled = norm_coords * y_std[:2] + y_mean[:2]
    return unscaled


def compute_ade(pred_coords, gt_coords):
    """
    Calculates Average Displacement Error (ADE).
    pred_coords: [N, T, 2] in physical units (meters or feet)
    gt_coords:   [N, T, 2] in physical units
    Returns: scalar float
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    # Euclidean distance per timestep per sample: sqrt((x - x_hat)^2 + (y - y_hat)^2)
    disp = np.linalg.norm(pred_coords - gt_coords, axis=-1)  # [N, T]
    return float(np.mean(disp))


def compute_fde(pred_coords, gt_coords):
    """
    Calculates Final Displacement Error (FDE) at final timestep T_pred.
    pred_coords: [N, T, 2] in physical units
    gt_coords:   [N, T, 2] in physical units
    Returns: scalar float
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    final_disp = np.linalg.norm(pred_coords[:, -1, :] - gt_coords[:, -1, :], axis=-1)  # [N]
    return float(np.mean(final_disp))


def compute_rmse(pred_coords, gt_coords):
    """
    Calculates Root Mean Square Error (RMSE) across coordinates.
    pred_coords: [N, T, 2]
    gt_coords:   [N, T, 2]
    Returns: scalar float
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    mse = np.mean((pred_coords - gt_coords) ** 2)
    return float(np.sqrt(mse))


def compute_horizon_ades(pred_coords, gt_coords, fps=2):
    """
    Computes cumulative ADE at each second mark (e.g. 1.0s, 2.0s, 3.0s).
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    T = pred_coords.shape[1]
    disp = np.linalg.norm(pred_coords - gt_coords, axis=-1)  # [N, T]
    
    horizon_results = {}
    for step in range(1, T + 1):
        sec = step / fps
        horizon_results[f"ADE_{sec:.1f}s"] = float(np.mean(disp[:, :step]))
        horizon_results[f"FDE_{sec:.1f}s"] = float(np.mean(disp[:, step-1]))
    
    return horizon_results


def compute_mae(pred_coords, gt_coords):
    """
    Calculates Mean Absolute Error (MAE) per coordinate, averaged across all dimensions.
    Formula: (1 / (N * T * D)) * sum(|pred - gt|)
    pred_coords: [N, T, D] or tensor
    gt_coords:   [N, T, D] or tensor
    Returns: scalar float
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    mae = np.mean(np.abs(pred_coords - gt_coords))
    return float(mae)


def compute_mape(pred_coords, gt_coords, epsilon=1.0):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Formula: (100 / (N * T * D)) * sum(|pred - gt| / max(|gt|, epsilon))
    pred_coords: [N, T, D] or tensor
    gt_coords:   [N, T, D] or tensor
    epsilon:     float, threshold to avoid division by zero near trajectory origin (default: 1.0m)
    Returns: scalar float
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    denom = np.maximum(np.abs(gt_coords), epsilon)
    mape = np.mean(np.abs(pred_coords - gt_coords) / denom) * 100.0
    return float(mape)


def compute_nade(pred_coords, gt_coords):
    """
    Calculates Normalized Average Displacement Error (NADE).
    NADE = ADE / mean(trajectory travel distance) * 100%
    pred_coords: [N, T, 2] in physical units
    gt_coords:   [N, T, 2] in physical units
    Returns: scalar float percentage
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    ade = compute_ade(pred_coords, gt_coords)
    step_diffs = np.linalg.norm(np.diff(gt_coords, axis=1), axis=-1)  # [N, T-1]
    first_step = np.linalg.norm(gt_coords[:, 0, :], axis=-1)          # [N]
    total_travel = first_step + np.sum(step_diffs, axis=-1)           # [N]
    mean_travel = float(np.mean(total_travel))
    
    if mean_travel < 1e-4:
        return 0.0
    return float((ade / mean_travel) * 100.0)


def compute_fairness_metrics(zone_errors_dict):
    """
    Computes fairness and equity metrics across spatial zones / clients.
    zone_errors_dict: dict of {zone_name: error_value}
    Returns: dict with:
        - worst_zone: tuple of (name, error) for highest error zone
        - best_zone: tuple of (name, error) for lowest error zone
        - mean_error: average error across zones
        - std_error: standard deviation across zones
        - coeff_variation: std / mean (relative disparity)
        - gap: worst_error - best_error
    """
    if not zone_errors_dict:
        return {
            'worst_zone': (None, 0.0),
            'best_zone': (None, 0.0),
            'mean_error': 0.0,
            'std_error': 0.0,
            'coeff_variation': 0.0,
            'gap': 0.0
        }

    worst_zone = max(zone_errors_dict.items(), key=lambda x: x[1])
    best_zone = min(zone_errors_dict.items(), key=lambda x: x[1])
    errors = list(zone_errors_dict.values())
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    coeff_variation = float(std_error / mean_error) if mean_error != 0.0 else 0.0
    gap = float(worst_zone[1] - best_zone[1])

    return {
        'worst_zone': (worst_zone[0], float(worst_zone[1])),
        'best_zone': (best_zone[0], float(best_zone[1])),
        'mean_error': mean_error,
        'std_error': std_error,
        'coeff_variation': coeff_variation,
        'gap': gap
    }


def compute_personalization_recovery(centralized_error, fedavg_error, pfl_error):
    """
    Computes personalization recovery rate and relative improvement against baselines.
    Returns: dict with:
        - fedavg_gap: fedavg_error - centralized_error
        - pfl_gap: pfl_error - centralized_error
        - recovery_pct: (fedavg_error - pfl_error) / (fedavg_error - centralized_error) * 100 (clamped 0-100)
        - improvement_pct: (fedavg_error - pfl_error) / fedavg_error * 100
    """
    fedavg_gap = float(fedavg_error - centralized_error)
    pfl_gap = float(pfl_error - centralized_error)

    denom = fedavg_error - centralized_error
    if denom != 0.0:
        recovery_pct = float(np.clip((fedavg_error - pfl_error) / denom * 100.0, 0.0, 100.0))
    else:
        recovery_pct = 0.0

    if fedavg_error != 0.0:
        improvement_pct = float((fedavg_error - pfl_error) / fedavg_error * 100.0)
    else:
        improvement_pct = 0.0

    return {
        'fedavg_gap': fedavg_gap,
        'pfl_gap': pfl_gap,
        'recovery_pct': recovery_pct,
        'improvement_pct': improvement_pct
    }


def evaluate_model_on_loader(model, dataloader, scalers, device="cpu"):
    """
    Runs model inference on a PyTorch DataLoader and computes all standardized metrics
    both in Feet (dataset native) and Meters (SI benchmark standard).
    """
    model.eval()
    all_preds_norm = []
    all_gts_raw = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, raw_targets = batch
            else:
                inputs, targets = batch
                raw_targets = targets

            inputs = inputs.to(device)
            preds = model(inputs)  # [B, T_pred, 2] (normalized)
            
            all_preds_norm.append(preds.cpu().numpy())
            all_gts_raw.append(raw_targets.cpu().numpy())

    preds_norm = np.concatenate(all_preds_norm, axis=0)
    gts_raw_feet = np.concatenate(all_gts_raw, axis=0)

    # Convert predictions back to feet
    preds_raw_feet = unscale_coordinates(preds_norm, scalers['y_mean'], scalers['y_std'])
    
    # Convert to meters
    preds_raw_meters = preds_raw_feet * FEET_TO_METERS
    gts_raw_meters = gts_raw_feet * FEET_TO_METERS

    # Compute metrics
    ade_meters = compute_ade(preds_raw_meters, gts_raw_meters)
    fde_meters = compute_fde(preds_raw_meters, gts_raw_meters)
    rmse_meters = compute_rmse(preds_raw_meters, gts_raw_meters)
    mae_meters = compute_mae(preds_raw_meters, gts_raw_meters)
    mape_meters = compute_mape(preds_raw_meters, gts_raw_meters)
    nade_pct = compute_nade(preds_raw_meters, gts_raw_meters)

    ade_feet = compute_ade(preds_raw_feet, gts_raw_feet)
    fde_feet = compute_fde(preds_raw_feet, gts_raw_feet)
    rmse_feet = compute_rmse(preds_raw_feet, gts_raw_feet)

    horizon_metrics = compute_horizon_ades(preds_raw_meters, gts_raw_meters)

    return {
        'ADE_m': ade_meters,
        'FDE_m': fde_meters,
        'RMSE_m': rmse_meters,
        'MAE_m': mae_meters,
        'MAPE_m': mape_meters,
        'NADE_pct': nade_pct,
        'ADE_ft': ade_feet,
        'FDE_ft': fde_feet,
        'RMSE_ft': rmse_feet,
        'horizons_m': horizon_metrics,
        'preds_raw_meters': preds_raw_meters,
        'gts_raw_meters': gts_raw_meters
    }


if __name__ == '__main__':
    # Test metric calculations with synthetic known values
    gt = np.zeros((100, 6, 2))
    pred = np.ones((100, 6, 2)) * 3.0  # Euclidean distance sqrt(9 + 9) = sqrt(18) = 4.2426
    
    ade = compute_ade(pred, gt)
    fde = compute_fde(pred, gt)
    rmse = compute_rmse(pred, gt)
    mae = compute_mae(pred, gt)
    mape = compute_mape(pred, gt)
    
    print(f"Synthetic Test - Expected ADE/FDE: ~4.2426 | Computed ADE: {ade:.4f}, FDE: {fde:.4f}, RMSE: {rmse:.4f}")
    print(f"Computed MAE: {mae:.4f}, MAPE: {mape:.4f}")
    assert abs(ade - 4.2426) < 1e-3, "ADE test failed"
    assert abs(fde - 4.2426) < 1e-3, "FDE test failed"
    assert abs(mae - 3.0) < 1e-3, "MAE test failed"

    # Test fairness metrics
    zone_errors = {'Zone_1': 1.2, 'Zone_2': 1.8, 'Zone_3': 2.4}
    fairness = compute_fairness_metrics(zone_errors)
    print("Fairness metrics:", fairness)
    assert fairness['worst_zone'] == ('Zone_3', 2.4), "Fairness worst_zone failed"
    assert fairness['best_zone'] == ('Zone_1', 1.2), "Fairness best_zone failed"
    assert abs(fairness['gap'] - 1.2) < 1e-4, "Fairness gap failed"

    # Test personalization recovery
    recovery = compute_personalization_recovery(centralized_error=1.0, fedavg_error=2.0, pfl_error=1.3)
    print("Recovery metrics:", recovery)
    assert abs(recovery['fedavg_gap'] - 1.0) < 1e-4
    assert abs(recovery['pfl_gap'] - 0.3) < 1e-4
    assert abs(recovery['recovery_pct'] - 70.0) < 1e-4
    assert abs(recovery['improvement_pct'] - 35.0) < 1e-4

    print("All metric unit tests passed successfully!")
