"""
Evaluation Metrics for Vehicle Trajectory Prediction.
Metrics implemented:
- Average Displacement Error (ADE) in meters & feet
- Final Displacement Error (FDE) in meters & feet
- Root Mean Square Error (RMSE)
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

    ade_feet = compute_ade(preds_raw_feet, gts_raw_feet)
    fde_feet = compute_fde(preds_raw_feet, gts_raw_feet)
    rmse_feet = compute_rmse(preds_raw_feet, gts_raw_feet)

    horizon_metrics = compute_horizon_ades(preds_raw_meters, gts_raw_meters)

    return {
        'ADE_m': ade_meters,
        'FDE_m': fde_meters,
        'RMSE_m': rmse_meters,
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
    
    print(f"Synthetic Test - Expected: ~4.2426 | Computed ADE: {ade:.4f}, FDE: {fde:.4f}, RMSE: {rmse:.4f}")
    assert abs(ade - 4.2426) < 1e-3, "ADE test failed"
    assert abs(fde - 4.2426) < 1e-3, "FDE test failed"
    print("All metric unit tests passed successfully!")
