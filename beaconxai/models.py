from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SklearnTSClassifier:
    model: object
    n_classes: int

    def logits(self, x: np.ndarray) -> np.ndarray:
        vec = x.reshape(1, -1)
        if hasattr(self.model, "decision_function"):
            out = self.model.decision_function(vec)
            if np.ndim(out) == 1:
                score = float(out[0])
                return np.array([-score, score], dtype=np.float64)
            return np.asarray(out[0], dtype=np.float64)

        probs = self.model.predict_proba(vec)[0]
        return np.log(np.clip(probs, 1e-12, 1.0))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))

    def margin_gradient(self, x: np.ndarray, y_hat: int | None = None) -> np.ndarray:
        """
        Gradient of adaptive margin wrt raw input x for linear-logit model.
        Only valid for StandardScaler + LogisticRegression pipeline.
        """
        if not hasattr(self.model, "named_steps"):
            raise RuntimeError("margin_gradient requires sklearn pipeline model")

        scaler = self.model.named_steps.get("standardscaler")
        lr = self.model.named_steps.get("logisticregression")
        if scaler is None or lr is None:
            raise RuntimeError("margin_gradient requires StandardScaler + LogisticRegression")

        logits = self.logits(x)
        y = int(np.argmax(logits)) if y_hat is None else int(y_hat)
        alt = int(np.argmax(np.where(np.arange(len(logits)) == y, -np.inf, logits)))

        w = lr.coef_  # [C, F]
        sigma = np.asarray(scaler.scale_, dtype=np.float64)
        sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)

        grad_flat = (w[y] - w[alt]) / sigma
        return grad_flat.reshape(x.shape)


def _ts_stat_features(x: np.ndarray) -> np.ndarray:
    """
    Build compact statistical features from raw [T, C] or batch [N, T, C] windows.
    """
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False

    feats = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.median(x, axis=1),
        (x**2).mean(axis=1),
        (np.diff(x, axis=1) ** 2).mean(axis=1),
    ]
    out = np.concatenate(feats, axis=1)
    return out[0] if squeeze else out


def _anfis_features(x: np.ndarray) -> np.ndarray:
    """
    Rich but fast HAR feature set for ANFIS.
    Returns [N, F] or [F] for single sample.
    """
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False

    n, t, c = x.shape
    base = _ts_stat_features(x)
    mean_abs = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    q25 = np.quantile(x, 0.25, axis=1)
    q75 = np.quantile(x, 0.75, axis=1)
    iqr = q75 - q25

    # Signal magnitude area-like summary
    sma = np.sum(np.abs(x), axis=1) / float(t)

    # Pairwise channel correlations per window
    xc = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1) + 1e-8
    corrs = []
    for i in range(c):
        for j in range(i + 1, c):
            num = np.mean(xc[:, :, i] * xc[:, :, j], axis=1)
            den = std[:, i] * std[:, j]
            corrs.append((num / den)[:, None])
    corr_feat = np.concatenate(corrs, axis=1) if corrs else np.zeros((n, 0), dtype=x.dtype)

    # Frequency-domain energy + entropy + dominant bin
    fx = np.fft.rfft(x, axis=1)
    pwr = (fx.real * fx.real + fx.imag * fx.imag).astype(np.float64, copy=False)
    pwr_nd = pwr[:, 1:, :] if pwr.shape[1] > 1 else pwr
    spec_energy = np.mean(pwr_nd, axis=1)
    ps = pwr_nd + 1e-12
    ps = ps / np.sum(ps, axis=1, keepdims=True)
    spec_entropy = -np.sum(ps * np.log(ps), axis=1) / np.log(ps.shape[1] + 1e-12)
    dom_bin = np.argmax(pwr_nd, axis=1).astype(np.float64) / float(max(1, pwr_nd.shape[1] - 1))

    out = np.concatenate(
        [base, mean_abs, rms, iqr, sma, corr_feat, spec_energy, spec_entropy, dom_bin],
        axis=1,
    )
    return out[0] if squeeze else out


@dataclass
class TreeStatsClassifier:
    model: ExtraTreesClassifier
    n_classes: int

    def logits(self, x: np.ndarray) -> np.ndarray:
        f = _ts_stat_features(x).reshape(1, -1)
        probs = self.model.predict_proba(f)[0]
        return np.log(np.clip(probs, 1e-12, 1.0))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))


@dataclass
class BoostingStatsClassifier:
    model: object
    n_classes: int

    def logits(self, x: np.ndarray) -> np.ndarray:
        f = _anfis_features(x).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(f)[0]
            return np.log(np.clip(probs, 1e-12, 1.0))
        if hasattr(self.model, "decision_function"):
            out = self.model.decision_function(f)
            if np.ndim(out) == 1:
                score = float(out[0])
                return np.array([-score, score], dtype=np.float64)
            return np.asarray(out[0], dtype=np.float64)
        raise RuntimeError("Boosting model has neither predict_proba nor decision_function")

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))


@dataclass
class AnfisStatsClassifier:
    feat_mean: np.ndarray
    feat_std: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    consequent_linear: np.ndarray
    consequent_bias: np.ndarray
    n_classes: int

    def _phi(self, x: np.ndarray) -> np.ndarray:
        f = _anfis_features(x).astype(np.float64, copy=False).reshape(1, -1)
        z = (f - self.feat_mean[None, :]) / self.feat_std[None, :]
        diff = (z[:, None, :] - self.centers[None, :, :]) / self.scales[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        w = np.exp(-0.5 * d2)
        phi = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
        return phi

    def logits(self, x: np.ndarray) -> np.ndarray:
        f = _anfis_features(x).astype(np.float64, copy=False).reshape(1, -1)
        z = (f - self.feat_mean[None, :]) / self.feat_std[None, :]
        phi = self._phi(x)
        # First-order Sugeno consequents:
        # f_r,c(z) = a_{r,c}^T z + b_{r,c}, y_c = sum_r phi_r * f_r,c
        rule_logits = np.einsum("nf,rfc->nrc", z, self.consequent_linear) + self.consequent_bias[None, :, :]
        out = np.sum(phi[:, :, None] * rule_logits, axis=1)
        return out[0].astype(np.float64, copy=False)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))


class _CNN1D(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.block1 = _ResBlock(64, 128, kernel_size=5, dropout=0.10)
        self.block2 = _ResBlock(128, 128, kernel_size=5, dropout=0.15)
        self.block3 = _ResBlock(128, 192, kernel_size=3, dropout=0.20)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(192, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm1d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.drop(z)
        return self.act(z + self.skip(x))


@dataclass
class TorchTSClassifier:
    model: nn.Module
    device: torch.device
    n_classes: int
    tta_shifts: tuple[int, ...] = (0,)

    def logits(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = []
            for s in self.tta_shifts:
                if s == 0:
                    z = xt
                else:
                    z = torch.roll(xt, shifts=int(s), dims=1)
                logits.append(self.model(z)[0])
            out = torch.stack(logits, dim=0).mean(dim=0).detach().cpu().numpy().astype(np.float64)
        return out

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))

    def margin_gradient(self, x: np.ndarray, y_hat: int | None = None) -> np.ndarray:
        self.model.eval()
        xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
        xt.requires_grad_(True)
        logits = self.model(xt)[0]
        y = int(torch.argmax(logits).item()) if y_hat is None else int(y_hat)

        tmp = logits.detach().clone()
        tmp[y] = -float("inf")
        alt = int(torch.argmax(tmp).item())

        margin = logits[y] - logits[alt]
        self.model.zero_grad(set_to_none=True)
        if xt.grad is not None:
            xt.grad.zero_()
        margin.backward()
        grad = xt.grad[0].detach().cpu().numpy().astype(np.float64)
        return grad


def train_logreg(x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 2000) -> SklearnTSClassifier:
    x_flat = x_train.reshape(x_train.shape[0], -1)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=max_iter),
    )
    clf.fit(x_flat, y_train)
    n_classes = int(np.max(y_train)) + 1
    return SklearnTSClassifier(model=clf, n_classes=n_classes)


def train_extratrees_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 600,
    random_state: int = 42,
) -> TreeStatsClassifier:
    x_feat = _ts_stat_features(x_train)
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(x_feat, y_train)
    n_classes = int(np.max(y_train)) + 1
    return TreeStatsClassifier(model=clf, n_classes=n_classes)


def train_histgbt_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 220,
    learning_rate: float = 0.08,
    max_leaf_nodes: int = 63,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> BoostingStatsClassifier:
    x_feat = _anfis_features(x_train).astype(np.float32, copy=False)
    y = y_train.astype(np.int64, copy=False)
    n_classes = int(np.max(y)) + 1

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts < 1.0, 1.0, counts)
    cls_w = float(len(y)) / (float(n_classes) * counts)
    sample_weight = cls_w[y]

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    clf.fit(x_feat, y, sample_weight=sample_weight)
    return BoostingStatsClassifier(model=clf, n_classes=n_classes)


def train_anfis_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_rules: int = 10,
    ridge: float = 2e-1,
    max_fit_samples: int = 4000,
    random_state: int = 42,
) -> AnfisStatsClassifier:
    x_feat = _anfis_features(x_train).astype(np.float64, copy=False)
    y = y_train.astype(np.int64, copy=False)
    if max_fit_samples > 0 and len(x_feat) > max_fit_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x_feat), size=max_fit_samples, replace=False)
        x_feat = x_feat[idx]
        y = y[idx]
    n_classes = int(np.max(y)) + 1

    feat_mean = x_feat.mean(axis=0)
    feat_std = x_feat.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    z = (x_feat - feat_mean[None, :]) / feat_std[None, :]

    n_rules_eff = int(max(2, min(n_rules, len(z))))
    km = KMeans(n_clusters=n_rules_eff, random_state=random_state, n_init=5)
    labels = km.fit_predict(z)
    centers = km.cluster_centers_.astype(np.float64, copy=False)

    global_scale = np.std(z, axis=0)
    global_scale = np.where(global_scale < 0.15, 0.15, global_scale)
    scales = np.zeros_like(centers)
    for r in range(n_rules_eff):
        idx = labels == r
        if np.sum(idx) >= 3:
            s = np.std(z[idx], axis=0)
            scales[r] = np.where(s < 0.10, global_scale, s)
        else:
            scales[r] = global_scale

    diff = (z[:, None, :] - centers[None, :, :]) / scales[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    w = np.exp(-0.5 * d2)
    phi = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)

    # Weighted least squares on first-order Sugeno design matrix.
    # X = [phi_r * z, phi_r]_{r=1..R}
    n_samples, n_feat = z.shape
    lin = (phi[:, :, None] * z[:, None, :]).reshape(n_samples, n_rules_eff * n_feat)
    design = np.concatenate([lin, phi], axis=1)

    y_onehot = np.eye(n_classes, dtype=np.float64)[y]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts < 1.0, 1.0, counts)
    cls_w = float(len(y)) / (float(n_classes) * counts)
    sw = cls_w[y]

    xw = design * sw[:, None]
    a = design.T @ xw + ridge * np.eye(design.shape[1], dtype=np.float64)
    b = design.T @ (y_onehot * sw[:, None])
    theta = np.linalg.solve(a, b)
    consequent_linear = theta[: n_rules_eff * n_feat, :].reshape(n_rules_eff, n_feat, n_classes)
    consequent_bias = theta[n_rules_eff * n_feat :, :].reshape(n_rules_eff, n_classes)

    return AnfisStatsClassifier(
        feat_mean=feat_mean.astype(np.float64, copy=False),
        feat_std=feat_std.astype(np.float64, copy=False),
        centers=centers,
        scales=scales.astype(np.float64, copy=False),
        consequent_linear=consequent_linear.astype(np.float64, copy=False),
        consequent_bias=consequent_bias.astype(np.float64, copy=False),
        n_classes=n_classes,
    )


def train_minirocket_if_available(x_train: np.ndarray, y_train: np.ndarray):
    try:
        from sktime.classification.kernel_based import RocketClassifier
    except Exception as exc:
        raise RuntimeError("sktime is not installed; install sktime to use MiniRocket baseline") from exc

    clf = RocketClassifier(
        num_kernels=10_000,
        rocket_transform="minirocket",
        use_multivariate="yes",
        random_state=42,
    )
    # sktime expects panel as [N, C, T], while project tensors are [N, T, C].
    x_train_panel = np.transpose(x_train, (0, 2, 1))
    clf.fit(x_train_panel, y_train)

    class _Wrapper:
        def __init__(self, inner):
            self.inner = inner

        def logits(self, x: np.ndarray) -> np.ndarray:
            x_panel = np.transpose(x[None, :, :], (0, 2, 1))
            if hasattr(self.inner, "predict_proba"):
                probs = self.inner.predict_proba(x_panel)[0]
                return np.log(np.clip(probs, 1e-12, 1.0))
            if hasattr(self.inner, "decision_function"):
                out = self.inner.decision_function(x_panel)
                if np.ndim(out) == 1:
                    score = float(out[0])
                    return np.array([-score, score], dtype=np.float64)
                return np.asarray(out[0], dtype=np.float64)
            raise RuntimeError("Rocket model has neither predict_proba nor decision_function")

        def predict(self, x: np.ndarray) -> int:
            return int(np.argmax(self.logits(x)))

    return _Wrapper(clf)


def train_1dcnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_frac: float = 0.1,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 8,
    use_class_weights: bool = True,
    tta_shifts: tuple[int, ...] = (0,),
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(np.max(y_train)) + 1
    model = _CNN1D(in_channels=x_train.shape[-1], n_classes=n_classes).to(device)

    x_t = torch.from_numpy(x_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.int64))
    n = len(x_t)
    n_val = int(max(1, n * val_frac))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx = perm
        val_idx = perm[:1]

    ds_tr = TensorDataset(x_t[tr_idx], y_t[tr_idx])
    ds_val = TensorDataset(x_t[val_idx], y_t[val_idx])
    dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = None
    if use_class_weights:
        counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        inv = inv / inv.mean()
        class_weights = torch.from_numpy(inv.astype(np.float32)).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for _epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                val_losses.append(float(crit(logits, yb).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchTSClassifier(model=model, device=device, n_classes=n_classes, tta_shifts=tta_shifts)
