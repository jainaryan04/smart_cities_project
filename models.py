"""
Neural Network Architectures for Vehicle Trajectory Prediction.
Models:
1. Seq2Seq LSTM (Vectorized Encoder-Decoder baseline)
2. Attention-augmented Seq2Seq LSTM
3. Model serialization, cloning, and FL parameter extraction utilities
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OBS_STEPS, PRED_STEPS


class TrajectoryEncoder(nn.Module):
    """Vectorized LSTM Encoder for past trajectory sequence [B, T_obs, input_dim]."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(TrajectoryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        embedded = self.input_embed(x)  # [B, T_obs, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        return lstm_out, (h_n, c_n)


class Seq2SeqLSTM(nn.Module):
    """
    High-Performance Vectorized Seq2Seq LSTM Architecture for Vehicle Trajectory Prediction.
    Maps historical sequence of length T_obs (4 features: x_rel, y_rel, vel, acc)
    to future sequence of length T_pred (2 features: x_rel, y_rel) in fully vectorized C++ kernels.
    """
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, pred_steps=PRED_STEPS):
        super(Seq2SeqLSTM, self).__init__()
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers, dropout)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, targets=None):
        # x: [B, T_obs, input_dim]
        _, (h_n, c_n) = self.encoder(x)
        
        # Decoder input: repeat the last encoder hidden state across T_pred steps
        # [B, 1, hidden_dim] -> [B, T_pred, hidden_dim]
        dec_in = h_n[-1].unsqueeze(1).repeat(1, self.pred_steps, 1)
        dec_out, _ = self.decoder_lstm(dec_in, (h_n, c_n))
        
        # Parallel projection [B, T_pred, output_dim]
        predictions = self.fc_out(dec_out)
        return predictions


class BahdanauAttention(nn.Module):
    """Additive Attention mechanism over encoder hidden states."""
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(BahdanauAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys):
        # query: [B, 1, hidden_dim], keys: [B, T_obs, hidden_dim]
        scores = self.V(torch.tanh(self.W_q(query) + self.W_k(keys)))  # [B, T_obs, 1]
        weights = F.softmax(scores, dim=1)                             # [B, T_obs, 1]
        context = torch.sum(weights * keys, dim=1, keepdim=True)        # [B, 1, hidden_dim]
        return context, weights


class AttentionSeq2SeqLSTM(nn.Module):
    """
    Attention-Augmented Seq2Seq LSTM Architecture.
    Computes spatial-temporal attention weights over historical frames.
    """
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, pred_steps=PRED_STEPS):
        super(AttentionSeq2SeqLSTM, self).__init__()
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.attention = BahdanauAttention(hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, targets=None):
        enc_out, (h_n, c_n) = self.encoder(x)  # enc_out: [B, T_obs, H]
        query = h_n[-1].unsqueeze(1)          # [B, 1, H]
        context, _ = self.attention(query, enc_out)  # [B, 1, H]
        
        # Combined context & hidden state repeated over prediction steps
        comb = torch.cat([h_n[-1].unsqueeze(1), context], dim=-1).repeat(1, self.pred_steps, 1)  # [B, T_pred, 2H]
        dec_out, _ = self.decoder_lstm(comb, (h_n, c_n))
        predictions = self.fc_out(dec_out)
        return predictions


class ConstantVelocityBaseline(nn.Module):
    """
    Constant-Velocity (CV) Extrapolation Baseline.
    Assumes velocity at the last observed step continues constantly through the prediction horizon.
    """
    def __init__(self, pred_steps=PRED_STEPS, scalers=None):
        super(ConstantVelocityBaseline, self).__init__()
        self.pred_steps = pred_steps
        self.scalers = scalers

    def forward(self, x, targets=None):
        if self.scalers is not None:
            std = torch.tensor(self.scalers['x_std'][:2], device=x.device, dtype=torch.float32)
            mean = torch.tensor(self.scalers['x_mean'][:2], device=x.device, dtype=torch.float32)
            p0 = x[:, -1, :2] * std + mean
            p1 = x[:, -2, :2] * std + mean
            v = p0 - p1
            steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
            pred_feet = steps * v.unsqueeze(1)
            y_std = torch.tensor(self.scalers['y_std'][:2], device=x.device, dtype=torch.float32)
            y_mean = torch.tensor(self.scalers['y_mean'][:2], device=x.device, dtype=torch.float32)
            return (pred_feet - y_mean) / y_std
        else:
            v = x[:, -1, :2] - x[:, -2, :2]
            steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
            return steps * v.unsqueeze(1)


class ConstantAccelerationBaseline(nn.Module):
    """
    Constant-Acceleration (CA) Extrapolation Baseline.
    Assumes acceleration estimated over the last two observed intervals continues through the prediction horizon.
    """
    def __init__(self, pred_steps=PRED_STEPS, scalers=None):
        super(ConstantAccelerationBaseline, self).__init__()
        self.pred_steps = pred_steps
        self.scalers = scalers

    def forward(self, x, targets=None):
        if self.scalers is not None:
            std = torch.tensor(self.scalers['x_std'][:2], device=x.device, dtype=torch.float32)
            mean = torch.tensor(self.scalers['x_mean'][:2], device=x.device, dtype=torch.float32)
            p0 = x[:, -1, :2] * std + mean
            p1 = x[:, -2, :2] * std + mean
            p2 = x[:, -3, :2] * std + mean
            v = p0 - p1
            v_prev = p1 - p2
            a = v - v_prev
            steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
            pred_feet = steps * v.unsqueeze(1) + 0.5 * (steps ** 2) * a.unsqueeze(1)
            y_std = torch.tensor(self.scalers['y_std'][:2], device=x.device, dtype=torch.float32)
            y_mean = torch.tensor(self.scalers['y_mean'][:2], device=x.device, dtype=torch.float32)
            return (pred_feet - y_mean) / y_std
        else:
            v = x[:, -1, :2] - x[:, -2, :2]
            v_prev = x[:, -2, :2] - x[:, -3, :2]
            a = v - v_prev
            steps = torch.arange(1, self.pred_steps + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
            return steps * v.unsqueeze(1) + 0.5 * (steps ** 2) * a.unsqueeze(1)


class ZoneAwareSeq2SeqLSTM(nn.Module):
    """
    Centralized Zone-Aware Seq2Seq LSTM.
    Shares the LSTM encoder-decoder representation across all zones,
    with zone-specific output heads (centralized FedPer counterpart).
    """
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, pred_steps=PRED_STEPS, num_zones=3):
        super(ZoneAwareSeq2SeqLSTM, self).__init__()
        self.pred_steps = pred_steps
        self.num_zones = num_zones
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.zone_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ) for _ in range(num_zones)
        ])
        self.active_zone_id = 0

    def forward(self, x, targets=None, zone_id=None):
        zid = zone_id if zone_id is not None else self.active_zone_id
        _, (h_n, c_n) = self.encoder(x)
        dec_in = h_n[-1].unsqueeze(1).repeat(1, self.pred_steps, 1)
        dec_out, _ = self.decoder_lstm(dec_in, (h_n, c_n))
        return self.zone_heads[zid](dec_out)


# -------------------------------------------------------------
# Federated Learning Helper Functions
# -------------------------------------------------------------

def get_model_parameters(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}


def set_model_parameters(model, parameters_dict):
    model.load_state_dict(parameters_dict)


def clone_model(model):
    return copy.deepcopy(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_size_kb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1024.0


if __name__ == '__main__':
    model1 = Seq2SeqLSTM()
    model2 = AttentionSeq2SeqLSTM()
    dummy_input = torch.randn(16, OBS_STEPS, INPUT_DIM)
    
    out1 = model1(dummy_input)
    out2 = model2(dummy_input)
    
    print(f"Seq2SeqLSTM output shape: {out1.shape}, Params: {count_parameters(model1):,}, Size: {compute_model_size_kb(model1):.2f} KB")
    print(f"AttentionSeq2Seq output shape: {out2.shape}, Params: {count_parameters(model2):,}, Size: {compute_model_size_kb(model2):.2f} KB")
