#!/usr/bin/env python3
"""
Task 3: Regression experiments to predict hours_since_watering using audio + sensors

Models:
- cnn1d_audio: 1D CNN on waveform
- cnn1d_audio_sensor: 1D CNN + sensor early fusion
- cnn2d_mel: 2D CNN on mel-spectrogram
- cnn2d_mel_sensor: 2D CNN + sensor early fusion
- wavelet_cnn2d: 2D CNN on wavelet scalogram (cwt)
- transformer_spec: Transformer encoder on mel-spectrogram frames
- ensemble_mean: Simple mean ensemble of selected models

Notes:
- Drops derived/leaky features (e.g., days_since_watering, last_watering_time, water_stress, water_status, time_bin, normalized cols)
- Uses results_task3/plant_ultrasound_dataset_with_sensors.csv as input
"""
import os
import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pywt
import warnings

# Suppress deprecated cwt warning from SciPy if emitted by dependencies
warnings.filterwarnings("ignore", message=r".*scipy.signal.cwt is deprecated.*")
import librosa

# ----------------------------- Config -----------------------------
@dataclass
class TrainConfig:
    csv_path: str = str(Path(__file__).resolve().parents[1] / 'results_task3' / 'plant_ultrasound_dataset_with_sensors.csv')
    audio_root: str = str(Path(__file__).resolve().parents[1])  # repo root to resolve relative audio paths
    sample_rate: int = 22050
    duration: float = 2.0
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    device: str = 'cuda'  # 'auto' | 'cpu' | 'cuda'
    models: List[str] = None  # if None, run all

    def __post_init__(self):
        if self.models is None:
            self.models = [
                'cnn1d_audio', 'cnn1d_audio_sensor',
                'cnn2d_mel', 'cnn2d_mel_sensor',
                'wavelet_cnn2d', 'transformer_spec',
                'ensemble_mean'
            ]
        # Env overrides
        env_models = os.getenv('TASK3_MODELS')
        if env_models:
            self.models = [m.strip() for m in env_models.split(',') if m.strip()]
        env_epochs = os.getenv('TASK3_EPOCHS')
        if env_epochs and env_epochs.isdigit():
            self.epochs = int(env_epochs)
        env_bs = os.getenv('TASK3_BATCH_SIZE')
        if env_bs and env_bs.isdigit():
            self.batch_size = int(env_bs)
        env_dur = os.getenv('TASK3_DURATION')
        if env_dur:
            try:
                self.duration = float(env_dur)
            except ValueError:
                pass

# ----------------------------- Utils -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str) -> torch.device:
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_audio_path(path_str: str, audio_root: str) -> str:
    if pd.isna(path_str) or not isinstance(path_str, str):
        return ''
    # If already absolute and exists
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    # Try join with repo root
    abs1 = os.path.join(audio_root, path_str)
    if os.path.exists(abs1):
        return abs1
    # Try without 'data/' prefix if duplicated
    if path_str.startswith('data/'):
        abs2 = os.path.join(audio_root, path_str[5:])
        if os.path.exists(abs2):
            return abs2
    return abs1  # fallback; loading will handle failure


# ----------------------------- Data -----------------------------
DERIVED_COLS = set([
    'days_since_watering', 'last_watering_time', 'water_stress', 'water_status',
    'time_bin', 'hours_normalized', 'days_normalized', 'sowing', 'transplant',
    'treatment', 'session_dir'
])

SENSOR_EXCLUDE = set(['sensor_time', 'sensor_source'])


def load_dataset(cfg: TrainConfig) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(cfg.csv_path)
    # Build absolute path for audio
    df['full_path'] = df['audio_path'].apply(lambda p: resolve_audio_path(p, cfg.audio_root))
    # Target
    if 'hours_since_watering' not in df.columns:
        raise ValueError('Target hours_since_watering not found in CSV')
    # Sensor columns
    sensor_cols = [c for c in df.columns if c.startswith('sensor_') and c not in SENSOR_EXCLUDE]
    # Remove non-numeric sensor columns
    num_sensor_cols = []
    for c in sensor_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_sensor_cols.append(c)
        else:
            # try coerce
            co = pd.to_numeric(df[c], errors='coerce')
            if co.notna().any():
                df[c] = co
                num_sensor_cols.append(c)
    # Drop derived/leaky features from consideration (we won't feed them as inputs)
    # We'll retain only columns needed for id/audio path/channel/sensor features/target
    keep_cols = ['audio_path', 'full_path', 'channel', 'session', 'recording_time', 'hours_since_watering'] + num_sensor_cols
    df = df[keep_cols].copy()
    return df, num_sensor_cols


# ----------------------------- Checkpointing/Artifacts -----------------------------
def compute_signature(cfg: TrainConfig, df: pd.DataFrame) -> str:
    try:
        csv_mtime = int(os.path.getmtime(cfg.csv_path))
        csv_name = os.path.basename(cfg.csv_path)
    except Exception:
        csv_mtime = 0
        csv_name = 'dataset'
    key = f"{csv_name}-{csv_mtime}-N{len(df)}-sr{cfg.sample_rate}-dur{cfg.duration}-bs{cfg.batch_size}-ep{cfg.epochs}-seed{cfg.seed}"
    # sanitize
    return key.replace(' ', '_').replace(':', '-')


def get_artifact_dirs() -> Dict[str, Path]:
    base = Path(__file__).resolve().parents[1] / 'models'
    dirs = {
        'base': base,
        'artifacts': base / 'task3_artifacts',
        'metrics': base / 'task3_metrics',
        'preds': base / 'task3_preds',
        'progress': base / 'task3_progress',
    }
    for d in dirs.values():
        d.mkdir(exist_ok=True, parents=True)
    return dirs


def model_done(model_name: str, signature: str, dirs: Dict[str, Path]) -> bool:
    metrics_path = dirs['metrics'] / f"{model_name}_{signature}.json"
    weights_path = dirs['artifacts'] / f"{model_name}_{signature}_best.pt"
    return metrics_path.exists() and weights_path.exists()


def save_metrics(model_name: str, signature: str, metrics: dict, dirs: Dict[str, Path]):
    metrics_path = dirs['metrics'] / f"{model_name}_{signature}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_preds(model_name: str, signature: str, preds: np.ndarray, dirs: Dict[str, Path]):
    np.save(dirs['preds'] / f"{model_name}_{signature}_test_preds.npy", preds)


def load_saved_preds(model_name: str, signature: str, dirs: Dict[str, Path]) -> Optional[np.ndarray]:
    p = dirs['preds'] / f"{model_name}_{signature}_test_preds.npy"
    if p.exists():
        return np.load(p)
    return None


class AudioSensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sensor_cols: List[str], cfg: TrainConfig,
                 mode: str = 'waveform', use_sensors: bool = False, scaler: Optional[StandardScaler] = None,
                 augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.sensor_cols = sensor_cols
        self.cfg = cfg
        self.mode = mode  # 'waveform' | 'mel' | 'wavelet' | 'spec_seq'
        self.use_sensors = use_sensors
        self.scaler = scaler
        self.augment = augment
        self.expected_len = int(cfg.sample_rate * cfg.duration)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            y, _ = librosa.load(path, sr=self.cfg.sample_rate)
        except Exception:
            y = np.zeros(self.expected_len, dtype=np.float32)
        if len(y) < 1:
            y = np.zeros(self.expected_len, dtype=np.float32)
        # pad/trim
        if len(y) < self.expected_len:
            y = np.pad(y, (0, self.expected_len - len(y)), mode='constant')
        else:
            y = y[:self.expected_len]
        # simple augment
        if self.augment:
            if np.random.rand() < 0.3:
                # Add small noise
                y = y + 0.005 * np.random.randn(len(y)).astype(np.float32)
            if np.random.rand() < 0.3:
                # gain
                y = y * np.random.uniform(0.8, 1.2)
        # normalize to [-1,1]
        m = np.max(np.abs(y)) + 1e-8
        y = y / m
        return y.astype(np.float32)

    def _mel(self, y: np.ndarray) -> np.ndarray:
        S = librosa.feature.melspectrogram(y=y, sr=self.cfg.sample_rate, n_fft=1024, hop_length=512, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)

    def _wavelet(self, y: np.ndarray) -> np.ndarray:
        # Continuous wavelet transform using PyWavelets (mexican hat 'mexh')
        # Memory mitigation: decimate signal and limit scales
        y_dec = y[::4]
        widths = np.arange(1, 64)
        cwt_mat, _freqs = pywt.cwt(y_dec, scales=widths, wavelet='mexh')  # (len(scales), len(y_dec))
        # normalize
        cwt_mat = (cwt_mat - cwt_mat.mean()) / (cwt_mat.std() + 1e-6)
        return cwt_mat.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['full_path']
        target = float(row['hours_since_watering'])
        y = self._load_audio(path)

        item: Dict[str, torch.Tensor] = {}
        if self.mode == 'waveform':
            item['audio'] = torch.from_numpy(y).unsqueeze(0)  # (1, T)
        elif self.mode == 'mel' or self.mode == 'spec_seq':
            mel = self._mel(y)  # (n_mels, time)
            item['spec'] = torch.from_numpy(mel).unsqueeze(0)  # (1, M, T)
        elif self.mode == 'wavelet':
            wavl = self._wavelet(y)  # (W, T)
            item['spec'] = torch.from_numpy(wavl).unsqueeze(0)  # (1, W, T)
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        if self.use_sensors:
            s = row[self.sensor_cols].values.astype(np.float32)
            if self.scaler is not None:
                s = self.scaler.transform([s])[0].astype(np.float32)
            # Asegura float32 para evitar errores de dtype
            item['sensor'] = torch.from_numpy(s).float()

        item['y'] = torch.tensor(target, dtype=torch.float32)
        return item


# ----------------------------- Models -----------------------------
class CNN1DEncoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # (B, 512)


class CNN2DEncoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # (B, out_dim)


class SpecTransformerEncoder(nn.Module):
    def __init__(self, n_mels=128, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = d_model

    def forward(self, spec):
        # spec: (B, 1, M, T)
        B, _, M, T = spec.shape
        x = spec.squeeze(1).transpose(1, 2)  # (B, T, M)
        x = self.proj(x)  # (B, T, d_model)
        x = self.encoder(x)  # (B, T, d_model)
        # pool over time
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return x


class RegressorHead(nn.Module):
    def __init__(self, in_dim: int, sensor_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        total = in_dim + sensor_dim
        self.mlp = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x, sensor=None):
        if sensor is not None:
            x = torch.cat([x, sensor], dim=1)
        return self.mlp(x).squeeze(1)


class AudioSensorRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, enc_out_dim: int, sensor_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.head = RegressorHead(enc_out_dim, sensor_dim)

    def forward(self, batch: dict):
        if 'audio' in batch:
            feats = self.encoder(batch['audio'])
        else:
            feats = self.encoder(batch['spec'])
        sensor = batch.get('sensor', None)
        return self.head(feats, sensor)


# ----------------------------- Training/Eval -----------------------------
@torch.no_grad()
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R^2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    # Pearson r
    if len(y_true) > 1:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 0.0
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': corr}


def train_one(cfg: TrainConfig, model_name: str, df: pd.DataFrame, sensor_cols: List[str], device: torch.device,
              out_dir: Path, signature: str, dirs: Dict[str, Path]) -> Tuple[dict, Optional[np.ndarray]]:
    # Select mode and sensor usage
    if model_name == 'cnn1d_audio':
        mode, use_sensors = 'waveform', False
    elif model_name == 'cnn1d_audio_sensor':
        mode, use_sensors = 'waveform', True
    elif model_name == 'cnn2d_mel':
        mode, use_sensors = 'mel', False
    elif model_name == 'cnn2d_mel_sensor':
        mode, use_sensors = 'mel', True
    elif model_name == 'wavelet_cnn2d':
        mode, use_sensors = 'wavelet', False
    elif model_name == 'transformer_spec':
        mode, use_sensors = 'spec_seq', True  # use sensors by default here
    else:
        raise ValueError(f'Unknown model {model_name}')

    # Split
    tr_df, te_df = train_test_split(df, test_size=0.2, random_state=cfg.seed)
    tr_df, va_df = train_test_split(tr_df, test_size=0.2, random_state=cfg.seed)

    # Sensor scaler
    scaler = None
    if use_sensors and sensor_cols:
        scaler = StandardScaler()
        scaler.fit(tr_df[sensor_cols].fillna(tr_df[sensor_cols].median()).to_numpy())
    
    # Datasets
    ds_tr = AudioSensorDataset(tr_df.fillna(tr_df.median(numeric_only=True)), sensor_cols, cfg, mode, use_sensors, scaler, augment=True)
    ds_va = AudioSensorDataset(va_df.fillna(va_df.median(numeric_only=True)), sensor_cols, cfg, mode, use_sensors, scaler, augment=False)
    ds_te = AudioSensorDataset(te_df.fillna(te_df.median(numeric_only=True)), sensor_cols, cfg, mode, use_sensors, scaler, augment=False)

    # Use smaller batch size for wavelet to avoid OOM
    batch_size = cfg.batch_size if model_name != 'wavelet_cnn2d' else max(2, min(8, cfg.batch_size // 2))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    if mode == 'waveform':
        enc = CNN1DEncoder()
        enc_out = 512
    elif mode in ['mel', 'wavelet']:
        enc = CNN2DEncoder()
        enc_out = enc.out_dim
    elif mode == 'spec_seq':
        enc = SpecTransformerEncoder(n_mels=128)
        enc_out = enc.out_dim
    else:
        raise ValueError('invalid mode')

    sensor_dim = len(sensor_cols) if use_sensors else 0
    model = AudioSensorRegressor(enc, enc_out, sensor_dim).to(device)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    loss_fn = nn.MSELoss()

    best_va = math.inf
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for batch in dl_tr:
            for k in ['audio', 'spec', 'sensor', 'y']:
                if k in batch:
                    batch[k] = batch[k].to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch['y'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        model.eval()
        with torch.no_grad():
            yv, pv = [], []
            for batch in dl_va:
                for k in ['audio', 'spec', 'sensor', 'y']:
                    if k in batch:
                        batch[k] = batch[k].to(device)
                pred = model(batch)
                yv.append(batch['y'].cpu().numpy())
                pv.append(pred.cpu().numpy())
            yv = np.concatenate(yv); pv = np.concatenate(pv)
            mae = np.mean(np.abs(yv - pv))
        if mae < best_va:
            best_va = mae
            # save best weights (both run-specific and signature global)
            best_run_path = out_dir / f'{model_name}_best.pt'
            torch.save(model.state_dict(), best_run_path)
            # also copy to artifacts dir with signature for resume/skip
            sig_path = dirs['artifacts'] / f"{model_name}_{signature}_best.pt"
            torch.save(model.state_dict(), sig_path)
        scheduler.step()
        print(f'[{model_name}] Epoch {epoch+1}/{cfg.epochs} TrainLoss={running/max(1,len(dl_tr)):.4f} ValMAE={mae:.3f}')

    # Test metrics
    # Load best
    best_path = out_dir / f'{model_name}_best.pt'
    if not best_path.exists():
        # try signature path (resume scenario)
        sig_path = dirs['artifacts'] / f"{model_name}_{signature}_best.pt"
        if sig_path.exists():
            model.load_state_dict(torch.load(sig_path, map_location=device))
        else:
            model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    yt, pt = [], []
    with torch.no_grad():
        for batch in dl_te:
            for k in ['audio', 'spec', 'sensor', 'y']:
                if k in batch:
                    batch[k] = batch[k].to(device)
            pred = model(batch)
            yt.append(batch['y'].cpu().numpy())
            pt.append(pred.cpu().numpy())
    yt = np.concatenate(yt); pt = np.concatenate(pt)
    metrics = regression_metrics(yt, pt)
    # save intermediate artifacts
    save_metrics(model_name, signature, metrics, dirs)
    save_preds(model_name, signature, pt, dirs)
    return metrics, pt


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    print('Loading dataset...')
    df, sensor_cols = load_dataset(cfg)
    print(f'Dataset size: {len(df)} | Sensor cols: {len(sensor_cols)}')
    signature = compute_signature(cfg, df)
    dirs = get_artifact_dirs()
    print(f'Run signature: {signature}')

    # Output dirs
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = models_dir / f'task3_{timestamp}'
    out_dir.mkdir(exist_ok=True)

    results = {}
    preds_store = {}

    # Train models (except ensemble)
    base_models = [m for m in cfg.models if m != 'ensemble_mean']
    for m in base_models:
        print(f'\n=== {m} ===')
        try:
            if model_done(m, signature, dirs):
                print(f'Skipping training for {m} (already done for signature).')
                # Load metrics
                with open(dirs['metrics'] / f"{m}_{signature}.json", 'r') as f:
                    results[m] = json.load(f)
                # Load preds for ensemble
                saved = load_saved_preds(m, signature, dirs)
                if saved is not None:
                    preds_store[m] = saved
                continue
            metrics, preds = train_one(cfg, m, df, sensor_cols, device, out_dir, signature, dirs)
            results[m] = metrics
            preds_store[m] = preds
            print(f'Results {m}: {metrics}')
        except Exception as e:
            print(f'Error in {m}: {e}')
        # Save incremental progress after each model
        try:
            progress_json = dirs['progress'] / f'progress_{signature}.json'
            with open(progress_json, 'w') as f:
                json.dump(results, f, indent=2)
            progress_csv = dirs['progress'] / f'progress_{signature}.csv'
            comp_rows = []
            for name, mres in results.items():
                row = {'model': name}
                row.update(mres)
                comp_rows.append(row)
            pd.DataFrame(comp_rows).to_csv(progress_csv, index=False)
        except Exception as e:
            print(f'Warning: could not save incremental progress: {e}')

    # Ensemble
    if 'ensemble_mean' in cfg.models:
        # Ensure preds available: if some models skipped, load from disk
        candidates = [m for m in base_models if model_done(m, signature, dirs)]
        for m in candidates:
            if m not in preds_store:
                p = load_saved_preds(m, signature, dirs)
                if p is not None:
                    preds_store[m] = p
        if preds_store:
            # Recompute test split to align with saved preds
            _, te_df = train_test_split(df, test_size=0.2, random_state=cfg.seed)
            pred_matrix = np.vstack([preds_store[m] for m in preds_store]).T  # (N, M)
            ensemble_pred = pred_matrix.mean(axis=1)
            yt = te_df['hours_since_watering'].values.astype(np.float32)
            ens_metrics = regression_metrics(yt, ensemble_pred)
            results['ensemble_mean'] = ens_metrics
            # Save ensemble predictions
            np.save(out_dir / f'ensemble_mean_{signature}.npy', ensemble_pred)

    # Save summary
    summary_path = models_dir / f'task3_regression_results_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    # CSV comparison
    comp_rows = []
    for name, m in results.items():
        row = {'model': name}
        row.update(m)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(models_dir / f'task3_final_comparison_{timestamp}.csv', index=False)

    print('\nCompleted experiments. Summary:')
    print(comp_df.sort_values('mae'))
    print(f'Artifacts in: {out_dir}')


if __name__ == '__main__':
    main()
