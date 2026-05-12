from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


class _CNN1D(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h)


@dataclass
class TorchTSClassifier:
    model: nn.Module
    device: torch.device
    n_classes: int

    def logits(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
            out = self.model(xt)[0].detach().cpu().numpy().astype(np.float64)
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


def train_minirocket_if_available(x_train: np.ndarray, y_train: np.ndarray):
    try:
        from sktime.classification.kernel_based import RocketClassifier
    except Exception as exc:
        raise RuntimeError("sktime is not installed; install sktime to use MiniRocket baseline") from exc

    clf = RocketClassifier(num_kernels=10_000, random_state=42)
    clf.fit(x_train, y_train)

    class _Wrapper:
        def __init__(self, inner):
            self.inner = inner

        def logits(self, x: np.ndarray) -> np.ndarray:
            probs = self.inner.predict_proba(x[None, :, :])[0]
            return np.log(np.clip(probs, 1e-12, 1.0))

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
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(np.max(y_train)) + 1
    model = _CNN1D(in_channels=x_train.shape[-1], n_classes=n_classes).to(device)

    ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    return TorchTSClassifier(model=model, device=device, n_classes=n_classes)
