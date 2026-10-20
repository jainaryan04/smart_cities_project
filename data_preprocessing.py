"""
Data Preprocessing Pipeline for NGSIM Vehicle Trajectory Dataset.
Features:
- Fast chunk-based parsing and sanitization of raw CSV
- Trajectory extraction with chronological frame ordering
- 2 Hz Downsampling and Sliding Window Sequence Generation
- Translation-invariant coordinate transformation
- Standard scaling of kinematic features
- Vehicle-level Train / Val / Test splitting
- Non-IID Federated Client Partitioning (US-101, Lankershim, Peachtree)
- Fast Caching with .npz files for instant re-loads
- PyTorch Dataset & DataLoader utilities
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_CSV, CACHE_DIR, OUTPUT_DIR, OBS_STEPS, PRED_STEPS, TOTAL_STEPS,
    DOWNSAMPLE_STEP, STRIDE, RANDOM_SEED, BATCH_SIZE, MAX_VEHICLES_PER_LOC,
    MAX_RAW_ROWS, NATIVE_FRAME_MS, MAX_FRAME_JUMP_FT
)


def assemble_tracks(loc_df, time_col=None):
    """
    Splits one location's rows into physically contiguous 10 Hz tracks.

    NGSIM Vehicle_IDs are reused across recording periods, and the raw export can
    have missing frames, so grouping by Vehicle_ID alone can join different vehicles
    into one track. A new track starts whenever the next row of the same Vehicle_ID
    is not the next 10 Hz frame (Frame_ID step != 1, or Global_Time step !=
    NATIVE_FRAME_MS when available) or jumps more than MAX_FRAME_JUMP_FT.

    Returns ([(vehicle_id, track_df), ...], stats_dict).
    """
    order_col = time_col if time_col else 'Frame_ID'
    df = loc_df.sort_values(['Vehicle_ID', order_col, 'Frame_ID'], kind='mergesort')
    n_before = len(df)
    df = df.drop_duplicates(subset=['Vehicle_ID', order_col])
    duplicates = n_before - len(df)

    same_vid = df['Vehicle_ID'].eq(df['Vehicle_ID'].shift())
    step_ok = df['Frame_ID'].diff().eq(1)
    if time_col:
        step_ok &= df[time_col].diff().eq(NATIVE_FRAME_MS)
    jump = np.hypot(df['Local_X'].diff(), df['Local_Y'].diff())
    jump_ok = jump.le(MAX_FRAME_JUMP_FT)

    frame_breaks = int((same_vid & ~step_ok).sum())
    jump_breaks = int((same_vid & step_ok & ~jump_ok).sum())
    new_track = ~(same_vid & step_ok & jump_ok)
    track_ids = new_track.cumsum()

    tracks = [(vid_track[0], g) for vid_track, g in df.groupby([df['Vehicle_ID'], track_ids], sort=False)]
    stats = {'rows': int(len(df)), 'tracks': len(tracks), 'frame_breaks': frame_breaks,
             'jump_breaks': jump_breaks, 'duplicates': int(duplicates)}
    return tracks, stats

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Vehicle Trajectory Sequences."""
    def __init__(self, inputs, targets, raw_targets=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.raw_targets = raw_targets  # Unscaled targets in feet for exact physical ADE/FDE

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.raw_targets is not None:
            return self.inputs[idx], self.targets[idx], torch.tensor(self.raw_targets[idx], dtype=torch.float32)
        return self.inputs[idx], self.targets[idx]


class NGSIMPreprocessor:
    """End-to-end Preprocessor for NGSIM Trajectory Data."""
    def __init__(self, csv_path=DATA_CSV, cache_dir=CACHE_DIR):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        # v2: contiguous-track assembly. Horizon lengths are part of the name so caches never mix.
        self.cache_file = os.path.join(cache_dir, f"federated_trajectory_data_v2_obs{OBS_STEPS}_pred{PRED_STEPS}.npz")
        self.scalers = {}

    def extract_sequences_from_csv(self, max_vehicles_per_loc=MAX_VEHICLES_PER_LOC, max_raw_rows=MAX_RAW_ROWS):
        """
        Reads CSV, sanitizes columns, extracts trajectories from a 1 Million row slice,
        assembles contiguous tracks, downsamples, and generates sliding window
        observation/prediction pairs.
        """
        print(f"[Preprocessor] Reading raw data from {self.csv_path} (limit: {max_raw_rows:,} rows)...", flush=True)
        t0 = time.time()

        header = pd.read_csv(self.csv_path, nrows=0).columns
        time_col = 'Global_Time' if 'Global_Time' in header else None
        if time_col is None:
            print("[Preprocessor] WARNING: no Global_Time column; recording periods are separated "
                  "only by Frame_ID gaps and position jumps.", flush=True)

        num_cols = ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc'] + ([time_col] if time_col else [])
        cols = num_cols + ['Location']
        loc_chunks = {'us-101': [], 'lankershim': [], 'peachtree': []}
        loc_counts = {'us-101': 0, 'lankershim': 0, 'peachtree': 0}
        target_per_loc = max_raw_rows // 3  # Balanced slice across 3 Smart City RSU domains
        
        for chunk in pd.read_csv(self.csv_path, usecols=cols, chunksize=500_000, low_memory=False):
            for col in num_cols:
                if chunk[col].dtype == object:
                    chunk[col] = pd.to_numeric(chunk[col].astype(str).str.replace(',', ''), errors='coerce')
            chunk = chunk.dropna(subset=cols)
            chunk['Location'] = chunk['Location'].astype(str).str.strip().str.lower()

            for loc_key in loc_chunks:
                if loc_counts[loc_key] < target_per_loc:
                    sub = chunk[chunk['Location'] == loc_key]
                    if not sub.empty:
                        needed = target_per_loc - loc_counts[loc_key]
                        sub_to_add = sub.iloc[:needed]
                        loc_chunks[loc_key].append(sub_to_add)
                        loc_counts[loc_key] += len(sub_to_add)

            if all(loc_counts[k] >= target_per_loc for k in loc_counts):
                break

        collected = []
        for loc_key, chunks_list in loc_chunks.items():
            if chunks_list:
                collected.append(pd.concat(chunks_list, ignore_index=True))
        
        df = pd.concat(collected, ignore_index=True)
        print(f"[Preprocessor] Ingested exactly {len(df):,} rows ({loc_counts}) in {time.time()-t0:.2f}s", flush=True)

        loc_order = ['us-101', 'lankershim', 'peachtree']
        loc_map = {
            'us-101': 0,
            'lankershim': 1,
            'peachtree': 2
        }

        client_data = {}
        split_rows = []
        for loc_name in loc_order:
            if loc_name not in df['Location'].values:
                continue
            cid = loc_map[loc_name]
            loc_df = df[df['Location'] == loc_name]
            unique_vids = loc_df['Vehicle_ID'].unique()
            
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(unique_vids)
            if max_vehicles_per_loc and len(unique_vids) > max_vehicles_per_loc:
                unique_vids = unique_vids[:max_vehicles_per_loc]

            print(f"[Preprocessor] Processing Location '{loc_name}' (Client {cid}) with {len(unique_vids)} vehicle IDs...", flush=True)
            
            n_v = len(unique_vids)
            n_train = int(0.70 * n_v)
            n_val = int(0.15 * n_v)
            
            # Splits are made by Vehicle_ID, so every track carrying the same ID
            # (even from different recording periods) stays in one split.
            train_vids = set(unique_vids[:n_train])
            val_vids = set(unique_vids[n_train:n_train+n_val])
            test_vids = set(unique_vids[n_train+n_val:])
            for vid in unique_vids:
                split_rows.append({'Location': loc_name, 'Vehicle_ID': int(vid),
                                   'Split': 'train' if vid in train_vids else ('val' if vid in val_vids else 'test')})

            sel_df = loc_df[loc_df['Vehicle_ID'].isin(set(unique_vids))]
            tracks, track_stats = assemble_tracks(sel_df, time_col=time_col)
            print(f"  -> {track_stats['rows']:,} rows -> {track_stats['tracks']:,} contiguous tracks "
                  f"({track_stats['frame_breaks']:,} frame/time breaks, {track_stats['jump_breaks']:,} position-jump breaks, "
                  f"{track_stats['duplicates']:,} duplicate frames dropped)", flush=True)

            splits = {'train': ([], []), 'val': ([], []), 'test': ([], [])}

            for vid, track in tracks:
                # Downsample within a contiguous track only, so consecutive samples are 0.5 s apart
                track = track.iloc[::DOWNSAMPLE_STEP]
                
                if len(track) < TOTAL_STEPS:
                    continue

                coords = track[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc']].values.astype(np.float32)
                
                if vid in train_vids:
                    split_key = 'train'
                elif vid in val_vids:
                    split_key = 'val'
                else:
                    split_key = 'test'

                for i in range(0, len(coords) - TOTAL_STEPS + 1, STRIDE):
                    window = coords[i : i + TOTAL_STEPS]
                    orig_x = window[OBS_STEPS - 1, 0]
                    orig_y = window[OBS_STEPS - 1, 1]

                    rel_window = window.copy()
                    rel_window[:, 0] -= orig_x
                    rel_window[:, 1] -= orig_y

                    inp_seq = rel_window[:OBS_STEPS]
                    tgt_seq = rel_window[OBS_STEPS:, :2]

                    splits[split_key][0].append(inp_seq)
                    splits[split_key][1].append(tgt_seq)

            client_data[cid] = {
                'name': loc_name,
                'train_x': np.array(splits['train'][0], dtype=np.float32).reshape(-1, OBS_STEPS, 4),
                'train_y': np.array(splits['train'][1], dtype=np.float32).reshape(-1, PRED_STEPS, 2),
                'val_x': np.array(splits['val'][0], dtype=np.float32).reshape(-1, OBS_STEPS, 4),
                'val_y': np.array(splits['val'][1], dtype=np.float32).reshape(-1, PRED_STEPS, 2),
                'test_x': np.array(splits['test'][0], dtype=np.float32).reshape(-1, OBS_STEPS, 4),
                'test_y': np.array(splits['test'][1], dtype=np.float32).reshape(-1, PRED_STEPS, 2),
                'raw_rows': np.array(track_stats['rows']),
            }
            print(f"  -> Client {cid} ({loc_name}): Train={len(client_data[cid]['train_x'])}, "
                  f"Val={len(client_data[cid]['val_x'])}, Test={len(client_data[cid]['test_x'])}", flush=True)

        split_path = os.path.join(OUTPUT_DIR, "data_split_vehicle_ids.csv")
        pd.DataFrame(split_rows).to_csv(split_path, index=False)
        print(f"[Preprocessor] Vehicle-level split written to {split_path}", flush=True)

        return client_data

    def fit_scalers_and_normalize(self, client_data):
        """Fits global standard scalers and normalizes features."""
        print("[Preprocessor] Computing feature normalization scalers...", flush=True)
        all_train_x = np.concatenate([client_data[cid]['train_x'] for cid in client_data], axis=0)
        
        flat_x = all_train_x.reshape(-1, all_train_x.shape[-1])
        x_mean = flat_x.mean(axis=0)
        x_std = flat_x.std(axis=0)
        x_std[x_std < 1e-6] = 1.0

        all_train_y = np.concatenate([client_data[cid]['train_y'] for cid in client_data], axis=0)
        flat_y = all_train_y.reshape(-1, all_train_y.shape[-1])
        y_mean = flat_y.mean(axis=0)
        y_std = flat_y.std(axis=0)
        y_std[y_std < 1e-6] = 1.0

        self.scalers = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

        print(f"  Input Mean: {x_mean}", flush=True)
        print(f"  Input Std:  {x_std}", flush=True)
        print(f"  Target Mean: {y_mean}", flush=True)
        print(f"  Target Std:  {y_std}", flush=True)

        normalized_data = {}
        for cid in client_data:
            normalized_data[cid] = {
                'name': client_data[cid]['name'],
                'train_x': (client_data[cid]['train_x'] - x_mean) / x_std,
                'train_y': (client_data[cid]['train_y'] - y_mean) / y_std,
                'val_x': (client_data[cid]['val_x'] - x_mean) / x_std,
                'val_y': (client_data[cid]['val_y'] - y_mean) / y_std,
                'test_x': (client_data[cid]['test_x'] - x_mean) / x_std,
                'test_y': (client_data[cid]['test_y'] - y_mean) / y_std,
                'raw_train_y': client_data[cid]['train_y'],
                'raw_val_y': client_data[cid]['val_y'],
                'raw_test_y': client_data[cid]['test_y'],
                'raw_rows': client_data[cid].get('raw_rows', np.array(0))
            }

        return normalized_data

    def save_cache(self, normalized_data):
        print(f"[Preprocessor] Saving preprocessed dataset to cache: {self.cache_file}", flush=True)
        save_dict = {
            'x_mean': self.scalers['x_mean'],
            'x_std': self.scalers['x_std'],
            'y_mean': self.scalers['y_mean'],
            'y_std': self.scalers['y_std'],
            'num_clients': len(normalized_data)
        }
        for cid, cdict in normalized_data.items():
            for key, val in cdict.items():
                if isinstance(val, np.ndarray):
                    save_dict[f"client_{cid}_{key}"] = val
                elif isinstance(val, str):
                    save_dict[f"client_{cid}_{key}"] = np.array(val)
        
        np.savez_compressed(self.cache_file, **save_dict)
        print("[Preprocessor] Cache successfully written.", flush=True)

    def load_cache(self):
        if not os.path.exists(self.cache_file):
            return None
        print(f"[Preprocessor] Loading preprocessed data from cache: {self.cache_file}", flush=True)
        data = np.load(self.cache_file, allow_pickle=True)
        self.scalers = {
            'x_mean': data['x_mean'],
            'x_std': data['x_std'],
            'y_mean': data['y_mean'],
            'y_std': data['y_std'],
        }
        num_clients = int(data['num_clients'])
        normalized_data = {}
        for cid in range(num_clients):
            name = str(data[f"client_{cid}_name"])
            normalized_data[cid] = {
                'name': name,
                'train_x': data[f"client_{cid}_train_x"],
                'train_y': data[f"client_{cid}_train_y"],
                'val_x': data[f"client_{cid}_val_x"],
                'val_y': data[f"client_{cid}_val_y"],
                'test_x': data[f"client_{cid}_test_x"],
                'test_y': data[f"client_{cid}_test_y"],
                'raw_train_y': data[f"client_{cid}_raw_train_y"],
                'raw_val_y': data[f"client_{cid}_raw_val_y"],
                'raw_test_y': data[f"client_{cid}_raw_test_y"],
                'raw_rows': data[f"client_{cid}_raw_rows"] if f"client_{cid}_raw_rows" in data else np.array(0),
            }
        print(f"[Preprocessor] Successfully loaded cached data for {num_clients} clients.", flush=True)
        return normalized_data

    def prepare_dataset(self, force_recompute=False):
        if not force_recompute:
            cached = self.load_cache()
            if cached is not None:
                return cached, self.scalers

        raw_client_data = self.extract_sequences_from_csv()
        normalized_data = self.fit_scalers_and_normalize(raw_client_data)
        self.save_cache(normalized_data)
        return normalized_data, self.scalers


def get_dataloaders(normalized_data, batch_size=BATCH_SIZE):
    client_loaders = {}
    
    all_train_x, all_train_y, all_raw_train_y = [], [], []
    all_val_x, all_val_y, all_raw_val_y = [], [], []
    all_test_x, all_test_y, all_raw_test_y = [], [], []

    for cid in sorted(normalized_data.keys()):
        cdict = normalized_data[cid]
        train_ds = TrajectoryDataset(cdict['train_x'], cdict['train_y'], cdict['raw_train_y'])
        val_ds = TrajectoryDataset(cdict['val_x'], cdict['val_y'], cdict['raw_val_y'])
        test_ds = TrajectoryDataset(cdict['test_x'], cdict['test_y'], cdict['raw_test_y'])

        client_loaders[cid] = {
            'name': cdict['name'],
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            'train_size': len(train_ds),
            'val_size': len(val_ds),
            'test_size': len(test_ds)
        }

        all_train_x.append(cdict['train_x'])
        all_train_y.append(cdict['train_y'])
        all_raw_train_y.append(cdict['raw_train_y'])

        all_val_x.append(cdict['val_x'])
        all_val_y.append(cdict['val_y'])
        all_raw_val_y.append(cdict['raw_val_y'])

        all_test_x.append(cdict['test_x'])
        all_test_y.append(cdict['test_y'])
        all_raw_test_y.append(cdict['raw_test_y'])

    cent_train_ds = TrajectoryDataset(
        np.concatenate(all_train_x, axis=0),
        np.concatenate(all_train_y, axis=0),
        np.concatenate(all_raw_train_y, axis=0)
    )
    cent_val_ds = TrajectoryDataset(
        np.concatenate(all_val_x, axis=0),
        np.concatenate(all_val_y, axis=0),
        np.concatenate(all_raw_val_y, axis=0)
    )
    cent_test_ds = TrajectoryDataset(
        np.concatenate(all_test_x, axis=0),
        np.concatenate(all_test_y, axis=0),
        np.concatenate(all_raw_test_y, axis=0)
    )

    centralized_loaders = {
        'train': DataLoader(cent_train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(cent_val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(cent_test_ds, batch_size=batch_size, shuffle=False),
        'train_size': len(cent_train_ds),
        'val_size': len(cent_val_ds),
        'test_size': len(cent_test_ds)
    }

    return client_loaders, centralized_loaders
