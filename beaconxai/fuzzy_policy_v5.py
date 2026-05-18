from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


EPS = 1e-8


FEATURES_V5 = [
    "m_neg",
    "M_B_minus",
    "r_B_minus",
    "CE_B",
    "rho_B_cost",
    "frag_drop",
    "top1_delta",
    "top3_sum_delta",
    "top3_conflict_count",
    "margin_entropy",
]


def _ensure_entropy(df):
    df = df.copy()
    if "delta_entropy" not in df.columns and "rank_entropy" in df.columns:
        df["delta_entropy"] = df["rank_entropy"]
    if "margin_entropy" not in df.columns:
        m = -df["m_neg"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        df["margin_entropy"] = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return df


def build_fuzzy_inputs_v5(df):
    d = _ensure_entropy(df)
    return d.loc[:, FEATURES_V5].to_numpy(dtype=np.float32)


class NeuroFuzzyV5(torch.nn.Module):
    def __init__(
        self,
        init_centers: np.ndarray,
        init_sigmas: np.ndarray,
        rule_terms: np.ndarray,
        init_rule_outputs: np.ndarray,
    ):
        super().__init__()
        self.n_features = int(init_centers.shape[0])
        self.n_terms = int(init_centers.shape[1])
        self.n_rules = int(rule_terms.shape[0])

        self.centers = torch.nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))
        self.log_sigmas = torch.nn.Parameter(torch.log(torch.tensor(np.clip(init_sigmas, 1e-3, None), dtype=torch.float32)))

        self.register_buffer("rule_terms", torch.tensor(rule_terms.astype(np.int64), dtype=torch.long))

        self.raw_rule_w = torch.nn.Parameter(torch.zeros(self.n_rules, dtype=torch.float32))
        init_out = np.clip(init_rule_outputs, 1e-4, 1.0 - 1e-4)
        self.raw_rule_out = torch.nn.Parameter(torch.tensor(np.log(init_out / (1.0 - init_out)), dtype=torch.float32))

        self.register_buffer("init_centers", torch.tensor(init_centers, dtype=torch.float32))
        self.register_buffer("init_log_sigmas", torch.log(torch.tensor(np.clip(init_sigmas, 1e-3, None), dtype=torch.float32)))

    def _membership(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,F] -> [N,F,T]
        sig = F.softplus(self.log_sigmas) + 1e-4
        z = (x.unsqueeze(-1) - self.centers.unsqueeze(0)) / sig.unsqueeze(0)
        return torch.exp(-0.5 * z * z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mf = self._membership(x)  # [N,F,T]
        n = mf.shape[0]
        r = self.n_rules
        f = self.n_features

        idx = self.rule_terms.view(1, r, f, 1).expand(n, r, f, 1)
        mf_e = mf.unsqueeze(1).expand(n, r, f, self.n_terms)
        chosen = torch.gather(mf_e, dim=3, index=idx).squeeze(-1)  # [N,R,F]
        acts = torch.prod(chosen, dim=2)  # [N,R]

        w = F.softplus(self.raw_rule_w) + 1e-4
        c = torch.sigmoid(self.raw_rule_out)

        num = torch.sum(acts * w.unsqueeze(0) * c.unsqueeze(0), dim=1)
        den = torch.sum(acts * w.unsqueeze(0), dim=1) + 1e-8
        return torch.clamp(num / den, 1e-6, 1.0 - 1e-6)

    def regularization(self, center_reg: float, sigma_reg: float, weight_reg: float) -> torch.Tensor:
        reg_c = center_reg * torch.mean((self.centers - self.init_centers) ** 2)
        reg_s = sigma_reg * torch.mean((self.log_sigmas - self.init_log_sigmas) ** 2)
        reg_w = weight_reg * torch.mean(self.raw_rule_w**2)
        return reg_c + reg_s + reg_w


@dataclass
class FuzzyV5Policy:
    model: NeuroFuzzyV5
    device: str
    features: list[str]


def _init_terms(X_train: np.ndarray, n_terms: int, seed: int):
    n_features = X_train.shape[1]
    centers = np.zeros((n_features, n_terms), dtype=np.float32)
    sigmas = np.zeros((n_features, n_terms), dtype=np.float32)

    for f in range(n_features):
        x = X_train[:, f : f + 1]
        km = KMeans(n_clusters=n_terms, random_state=seed, n_init=10)
        km.fit(x)
        c = np.sort(km.cluster_centers_.reshape(-1).astype(np.float32))
        centers[f] = c
        for t in range(n_terms):
            d = np.abs(x.reshape(-1) - c[t])
            sigmas[f, t] = float(max(np.mean(d), 1e-3))
    return centers, sigmas


def _init_rules(X_train: np.ndarray, y_train: np.ndarray, centers: np.ndarray, n_rules: int, seed: int):
    km = KMeans(n_clusters=n_rules, random_state=seed + 7, n_init=10)
    km.fit(X_train)
    c = km.cluster_centers_
    labels = km.labels_

    n_features = X_train.shape[1]
    n_terms = centers.shape[1]
    rule_terms = np.zeros((n_rules, n_features), dtype=np.int64)
    for r in range(n_rules):
        for f in range(n_features):
            idx = int(np.argmin(np.abs(centers[f] - c[r, f])))
            idx = max(0, min(n_terms - 1, idx))
            rule_terms[r, f] = idx

    out = np.zeros(n_rules, dtype=np.float32)
    for r in range(n_rules):
        m = labels == r
        out[r] = float(np.mean(y_train[m])) if np.any(m) else float(np.mean(y_train))
    return rule_terms, np.clip(out, 1e-3, 1.0 - 1e-3)


def fit_fuzzy_policy_v5(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_terms: int = 3,
    n_rules: int = 7,
    epochs: int = 350,
    lr: float = 3e-2,
    batch_size: int = 512,
    center_reg: float = 5e-4,
    sigma_reg: float = 5e-4,
    weight_reg: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> FuzzyV5Policy:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    centers, sigmas = _init_terms(X_train, n_terms=n_terms, seed=seed)
    rule_terms, rule_outputs = _init_rules(X_train, y_train, centers, n_rules=n_rules, seed=seed)

    model = NeuroFuzzyV5(centers, sigmas, rule_terms, rule_outputs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    patience = 50
    bad = 0

    n = X_train.shape[0]
    for _ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            xb = Xtr[b]
            yb = ytr[b]

            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            loss = loss + model.regularization(center_reg=center_reg, sigma_reg=sigma_reg, weight_reg=weight_reg)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        with torch.no_grad():
            pv = model(Xva)
            lv = F.binary_cross_entropy(pv, yva).item()
        if lv < best_val - 1e-5:
            best_val = lv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FuzzyV5Policy(model=model, device=device, features=FEATURES_V5)


def predict_fuzzy_policy_v5(policy: FuzzyV5Policy, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    mdl = policy.model
    mdl.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=policy.device)
            out.append(mdl(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)