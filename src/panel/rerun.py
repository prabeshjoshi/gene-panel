"""Rerun methods (R1) and helpers for R2/R3.

Every selector sees training rows only and picks from the CANDIDATE columns; scoring is on the
TARGET columns. Arms A/B: candidates = targets = all genes. Arm C: disjoint blocks.
The pilot's functions in panel.select stay unchanged; this module adds the corrected versions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.model_selection import KFold

from panel.evaluate import r2_all, r2_subset, ridge_coef
from panel.select import Standardizer


# ------------------------------------------------------------------------------- selectors

def greedy_targets(Z: np.ndarray, cand, targ, k: int, eps: float = 1e-10):
    """Greedy forward selection from candidates maximising training OLS R2 of the targets.

    Rank-one updates in gene space: Mcc = residual candidate Gram, Mct = residual
    candidate-target cross products. Adding candidate j removes ||Mct[j]||^2 / Mcc[j, j] of the
    targets' residual sum of squares. Returns (gene indices in selection order, R2 path).
    With cand = targ = all genes this is exactly the pilot's greedy_forward.
    """
    cand = np.asarray(cand)
    C, T = Z[:, cand], Z[:, np.asarray(targ)]
    Mcc, Mct = C.T @ C, C.T @ T
    total = float((T ** 2).sum())
    d0 = np.diag(Mcc).copy()
    rss, sel, path = total, [], []
    for _ in range(k):
        d = np.diag(Mcc)
        valid = d > eps * d0
        valid[sel] = False
        gains = np.full(len(cand), -np.inf)
        gains[valid] = (Mct[valid] ** 2).sum(axis=1) / d[valid]
        j = int(np.argmax(gains))
        mc, mt, dj = Mcc[:, j].copy(), Mct[j].copy(), Mcc[j, j]
        Mcc -= np.outer(mc, mc) / dj
        Mct -= np.outer(mc, mt) / dj
        rss -= gains[j]
        sel.append(j)
        path.append(1.0 - rss / total)
    return cand[sel], np.array(path)


def pivoted_qr_order(Z: np.ndarray, cand) -> np.ndarray:
    """Candidate genes in column-pivoted QR order (first k = the k-gene panel)."""
    _, _, piv = qr(Z[:, np.asarray(cand)], pivoting=True, mode="economic")
    return np.asarray(cand)[piv]


# ------------------------------------------------------------------------------- PLS2

def pls2_fit(X: np.ndarray, Y: np.ndarray, n_comp: int):
    """PLS2 (NIPALS model, regression-mode deflation, no rescaling) on centred X, Y.

    Each weight vector is the dominant eigenvector of X_a' (Y Y') X_a, which is what NIPALS
    converges to; it is found here with Lanczos (or a dense solve for small X), so no iteration
    cap applies. Returns dict W, T, P, Q, R (rotations: Yhat = X R[:, :a] Q[:, :a]').
    Components stop early if X is exhausted.
    """
    n, p = X.shape
    K = Y @ Y.T
    Xa = X.copy()
    W, T, P, Q = [], [], [], []
    scale0 = float((X ** 2).sum())
    for _ in range(min(n_comp, p, n - 1)):
        if p <= 200:
            A = Xa.T @ (K @ Xa)
            w = np.linalg.eigh((A + A.T) / 2.0)[1][:, -1]
        else:
            op = LinearOperator((p, p), matvec=lambda v: Xa.T @ (K @ (Xa @ v)), dtype=np.float64)
            w = eigsh(op, k=1, which="LA", v0=np.ones(p), tol=0)[1][:, 0]
        w = w / np.linalg.norm(w)
        t = Xa @ w
        tt = float(t @ t)
        if tt <= 1e-12 * scale0:
            break
        pa = Xa.T @ t / tt
        W.append(w); T.append(t); P.append(pa); Q.append(Y.T @ t / tt)
        Xa -= np.outer(t, pa)
    W, T, P, Q = (np.column_stack(m) for m in (W, T, P, Q))
    R = W @ np.linalg.inv(P.T @ W)
    return {"W": W, "T": T, "P": P, "Q": Q, "R": R}


def vip_scores(fit: dict, n_comp: int | None = None) -> np.ndarray:
    W, T, Q = fit["W"], fit["T"], fit["Q"]
    if n_comp is not None:
        W, T, Q = W[:, :n_comp], T[:, :n_comp], Q[:, :n_comp]
    ss = (T ** 2).sum(axis=0) * (Q ** 2).sum(axis=0)
    Wn = W / np.linalg.norm(W, axis=0)
    return np.sqrt(W.shape[0] * ((Wn ** 2) @ ss) / ss.sum())


def pls2_cv_lv(Z: np.ndarray, xcols, ycols, max_lv: int, n_folds: int, seed: int):
    """Number of LVs by K-fold CV on training rows (each fold re-standardised on its own
    training part). Score = pooled CV R2 over the Y columns. Returns (best, {lv: cv R2})."""
    xcols, ycols = np.asarray(xcols), np.asarray(ycols)
    max_lv = int(min(max_lv, len(xcols)))
    sse, ss = np.zeros(max_lv), 0.0
    for tr, va in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(Z):
        std = Standardizer.fit(Z[tr])
        Ztr, Zva = std.transform(Z[tr]), std.transform(Z[va])
        fit = pls2_fit(Ztr[:, xcols], Ztr[:, ycols], max_lv)
        Yva = Zva[:, ycols]
        ss += float((Yva ** 2).sum())
        S = Zva[:, xcols] @ fit["R"]
        for a in range(1, max_lv + 1):
            if a > S.shape[1]:
                sse[a - 1] = np.inf
                continue
            sse[a - 1] += float(((Yva - S[:, :a] @ fit["Q"][:, :a].T) ** 2).sum())
    cv = 1.0 - sse / ss
    return int(np.argmax(cv)) + 1, {a + 1: float(v) for a, v in enumerate(cv)}


def pls2_vip_iterative(Z: np.ndarray, cand, targ, ks, drop_fraction: float, max_lv: int,
                       n_folds: int, seed: int):
    """Backward elimination by PLS2 VIP: LVs by inner CV, drop the lowest-VIP
    floor(drop_fraction x remaining) candidates (at least 1), refit, repeat.

    One path from all candidates down to min(ks); each removal is capped so the path stops
    exactly at every k. Returns ({k: gene indices}, rounds info).
    """
    rem = np.asarray(cand).copy()
    targ = np.asarray(targ)
    stops = sorted({int(k) for k in ks}, reverse=True)
    panels, rounds = {}, []
    while len(rem) > stops[-1]:
        n_lv, _ = pls2_cv_lv(Z, rem, targ, max_lv, n_folds, seed)
        fit = pls2_fit(Z[:, rem], Z[:, targ], n_lv)
        vip = vip_scores(fit)
        nxt = max(k for k in stops if k < len(rem))
        d = min(max(1, int(np.floor(drop_fraction * len(rem)))), len(rem) - nxt)
        keep = np.sort(np.argsort(-vip, kind="stable")[: len(rem) - d])
        rounds.append({"remaining": int(len(rem)), "n_lv": int(n_lv), "dropped": int(d)})
        rem = rem[keep]
        if len(rem) in stops:
            panels[int(len(rem))] = rem.copy()
    return panels, rounds


# ------------------------------------------------------------------------------- scoring

def score_on_targets(Ztr, Zte, panel, targ, alpha: float, disjoint: bool) -> dict:
    """Ridge from panel genes to the target genes, fitted on training rows, scored on test.

    r2 = R2 over all targets (Arm C: the target block; Arms A/B: R2_all). Arms A/B also get
    q2_nonpanel = R2 over targets that are not in the panel.
    """
    panel, targ = np.asarray(panel), np.asarray(targ)
    B = ridge_coef(Ztr[:, panel], Ztr[:, targ], alpha)
    Yte = Zte[:, targ]
    Yhat = Zte[:, panel] @ B
    out = {"r2": r2_all(Yte, Yhat), "q2_nonpanel": np.nan}
    if not disjoint:
        nonpanel = np.flatnonzero(~np.isin(targ, panel))
        out["q2_nonpanel"] = r2_subset(Yte, Yhat, nonpanel)
    return out


def ols_rss(Z, panel, targ) -> float:
    """Training residual sum of squares of the targets after OLS on the panel."""
    S, T = Z[:, np.asarray(panel)], Z[:, np.asarray(targ)]
    G, H = S.T @ S, S.T @ T
    return float((T ** 2).sum() - np.trace(np.linalg.solve(G, H) @ H.T))


# ------------------------------------------------------------------------------- local search

def swap_local_search(Z: np.ndarray, cand, targ, start, max_iter: int, rel_tol: float = 1e-12):
    """Best-improvement 1-swap local search on the training OLS objective (targets' RSS).

    Every swap (panel gene i out, non-panel candidate j in) is scored exactly with low-rank
    updates: with Q = inv(G_SS), the residual of panel gene i on the rest of the panel is
    r_i = Z_S Q e_i / Q_ii (||r_i||^2 = 1/Q_ii), so dropping i adds ||T' r_i||^2 Q_ii to the RSS,
    and adding j then removes ||M_ct^{-i}[j]||^2 / M_cc^{-i}[j, j], with the residual
    cross-products M^{-i} = M^S + (C' r_i)(r_i' [C T]) Q_ii. Returns (panel, info).
    """
    cand = np.asarray(cand)
    C, T = Z[:, cand], Z[:, np.asarray(targ)]
    CT, cdiag, TT = C.T @ T, (C ** 2).sum(axis=0), float((T ** 2).sum())
    pos = {g: i for i, g in enumerate(cand)}
    S = np.array([pos[g] for g in start])
    history = []
    f = None
    for it in range(max_iter + 1):
        ZS = C[:, S]
        Q = np.linalg.inv(ZS.T @ ZS)
        CS, ST = C.T @ ZS, ZS.T @ T
        f = TT - float(np.trace(Q @ ST @ ST.T))
        history.append(f)
        if it == max_iter:
            break
        Mct = CT - CS @ (Q @ ST)
        Mdiag = cdiag - np.einsum("jk,kl,jl->j", CS, Q, CS)
        qd = np.diag(Q)
        A = (CS @ Q) / qd                    # c x k: C' r_i
        Bm = (ST.T @ Q) / qd                 # t x k: T' r_i
        s = 1.0 / qd                         # ||r_i||^2
        num = ((Mct ** 2).sum(axis=1)[:, None] + 2.0 * A * (Mct @ Bm) / s
               + A ** 2 * (Bm ** 2).sum(axis=0) / s ** 2)
        den = Mdiag[:, None] + A ** 2 / s
        ok = den > 1e-10 * cdiag[:, None]
        ok[S] = False
        new_f = np.where(ok, f + (Bm ** 2).sum(axis=0) / s - num / np.where(ok, den, 1.0), np.inf)
        j, i = np.unravel_index(np.argmin(new_f), new_f.shape)
        if not new_f[j, i] < f - rel_tol * TT:
            break
        S = S.copy()
        S[i] = j
    return cand[S], {"iterations": len(history) - 1, "rss_path": history,
                     "train_r2_start": 1 - history[0] / TT, "train_r2_end": 1 - history[-1] / TT}
