# file name: summary_stats_elms_260528.py

import numpy as np


def avg_div_numpy(SSPrev_obs) -> float:
    """
    Average Simpson's diversity index over time.
    SSPrev_obs: array of shape (42, 23)
    """
    Diversity_obs = np.zeros(SSPrev_obs.shape[1])  # length 23

    for j in range(SSPrev_obs.shape[1]):
        col = SSPrev_obs[:, j].astype(float)
        N = np.sum(col)
        denom = np.sum(col * (col - 1))
        if denom == 0:
            if N == 0:
                D = 0.0  # no infections → 0
            else:
                D = float(N)  # one of each strain → N
        else:
            D = N * (N - 1) / denom
        Diversity_obs[j] = D

    return float(np.mean(Diversity_obs))  #  consistent with MATLAB


def max_abundance_numpy(SSPrev_obs) -> float:
    """
    Peak abundance of a strain at any time.
    SSPrev_obs: array-like (n_strains, T). NaNs ignored.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    if X.size == 0 or np.isnan(X).all():
        return float("nan")
    return float(np.nanmax(X))


def avg_time_obs_str(SSPrev_obs: np.ndarray, *, nan_if_none: bool = True) -> float:
    """
    AvgTimeObsStr: average # of time points a *seen* strain is present (>0).

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.
    nan_if_none: if True, return np.nan when no strain is ever seen; else 0.0.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    # presence/absence per strain over time
    present = (X > 0).astype(np.int32)  # (S, T)
    per_strain_counts = present.sum(axis=1)  # (S,)
    seen = per_strain_counts[per_strain_counts > 0]
    if seen.size == 0:
        return float("nan") if nan_if_none else 0.0
    return float(seen.mean())


def max_time_obs_str(SSPrev_obs: np.ndarray, *, nan_if_none: bool = True) -> float:
    """
    MaxTimeObsStr: longest observed persistence (in time points) among all strains.

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.
    nan_if_none: if True, return np.nan when no strain is ever seen; else 0.0.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    present = (X > 0).astype(np.int32)  # (S, T) presence/absence
    per_strain_counts = present.sum(axis=1)  # time-points seen for each strain
    seen = per_strain_counts[per_strain_counts > 0]
    if seen.size == 0:
        return float("nan") if nan_if_none else 0.0
    return float(seen.max())


def num_strains_obs_str(SSPrev_obs: np.ndarray) -> int:
    """
    Count how many strains were ever observed (any nonzero across time).

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.

    Returns
    -------
    int
        Number of strains with at least one nonzero entry.
    """
    X = np.asarray(SSPrev_obs)
    return int(np.any(X > 0, axis=1).sum())


def var_div_numpy(SSPrev_obs) -> float:
    """
    Variance of Simpson's diversity index over time.
    SSPrev_obs: array of shape (42, 23)
    """
    Diversity_obs = np.zeros(SSPrev_obs.shape[1])  # length 23

    for j in range(SSPrev_obs.shape[1]):
        col = SSPrev_obs[:, j].astype(float)
        N = np.sum(col)
        denom = np.sum(col * (col - 1))
        if denom == 0:
            if N == 0:
                D = 0.0        # no infections → 0
            else:
                D = float(N)   # one of each strain → N
        else:
            D = N * (N - 1) / denom
        Diversity_obs[j] = D

    return float(np.var(Diversity_obs, ddof=1))  # ddof=1 consistent with MATLAB var()


def avg_time_repeat_inf_numpy(SSPrev_obs: np.ndarray, *, nan_if_empty: bool = True) -> float:
    """
    Average time between repeat detections across strains.
    SSPrev_obs: array of shape (42, 23)
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    T = X.shape[1]  # number of time points = 23
    E = []

    for i in range(X.shape[0]):  # for each strain
        # Find indices in flattened transposed matrix
        # MATLAB uses 1-based indexing, so we replicate the logic
        row = X[i, :]  # 1 × T
        # Find time point indices where strain is present
        vec = np.flatnonzero(row)  # 0-based indices

        if vec.size >= 2:
            gaps = np.diff(vec)        # gaps between observed time points
            gaps = gaps[gaps > 1]      # keep only gaps > 1
            gaps = gaps - 1            # subtract 1
            E.append(gaps)

    if not E or all(e.size == 0 for e in E):
        return float("nan") if nan_if_empty else 0.0

    gaps_all = np.concatenate(E).astype(float)
    return float(gaps_all.mean())  # consistent with MATLAB mean()


def var_time_repeat_inf_numpy(SSPrev_obs: np.ndarray, *, nan_if_empty: bool = True) -> float:
    """
    Variance of pooled gaps between repeat detections across strains.
    SSPrev_obs: array of shape (42, 23)
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    T = X.shape[1]  # 23 time points

    # find linear indices of non-zero in transposed matrix (0-based)
    A = np.flatnonzero(X.T > 0)

    E = []

    for i in range(1, X.shape[0] + 1):

        # extract indices belonging to strain i
        vec = A[(A >= (i-1)*T) & (A < i*T)]

        # compute gaps
        vec = np.diff(vec)

        # keep only gaps > 1
        vec = vec[vec > 1]

        # subtract 1
        vec = vec - 1

        if vec.size > 0:
            E.append(vec)

    if not E:
        return float("nan") if nan_if_empty else 0.0

    gaps = np.concatenate(E).astype(float)

    if gaps.size < 2:
        return float("nan") if nan_if_empty else 0.0

    return float(np.var(gaps, ddof=1))  # ddof=1 consistent with MATLAB var()


def avg_npmi_numpy(SSPrev_obs) -> float:
    """
    Average normalized PMI across *pairs of strains* (lower triangle).
    SSP: array-like, shape (S, T), nonnegative counts per strain (rows) over time (cols).
    Returns NaN if there are <2 strains or total count is zero.
    """
    SSP = np.asarray(SSPrev_obs, dtype=float)
    if SSP.ndim != 2:
        raise ValueError("SSP must be 2D (strains x time)")

    S, T = SSP.shape
    if S < 2:
        return np.nan

    TotalO = SSP.sum()
    if TotalO <= 0:
        return np.nan

    # Marginal probabilities P(X=i)
    PX = SSP.sum(axis=1) / TotalO  # shape (S,)

    # Pairwise joint probabilities P(X=i,Y=j) via sum_t min(SSP[i,t], SSP[j,t]) / TotalO
    # Build (S,S,T) of pairwise minima and sum over time (OK for S ~ 42, T modest)
    # mins[i,j,t] = min(SSP[i,t], SSP[j,t])
    A = SSP[:, None, :]            # (S,1,T)
    B = SSP[None, :, :]            # (1,S,T)
    mins = np.minimum(A, B)        # (S,S,T)
    PXY = mins.sum(axis=2) / TotalO  # (S,S)

    # Compute NPMI on the strict lower triangle (j < i)
    # NPMI(i,j) = (log PXY - log PX_i - log PX_j) / (-log PXY), if PXY>0
    # If PXY==0 => -1; if PX_i + PX_j == 0 => 0 (per original logic)
    i_idx, j_idx = np.tril_indices(S, k=-1)
    pxy = PXY[i_idx, j_idx]
    pxi = PX[i_idx]
    pxj = PX[j_idx]

    # Start with all -1 (the value when pxy == 0)
    npmi = np.full(pxy.shape, -1.0, dtype=float)

    # Valid where PX_i + PX_j > 0
    valid_marg = (pxi + pxj) > 0
    # Among those, where PXY > 0
    valid_joint = valid_marg & (pxy > 0)

    # Cases where PX_i + PX_j == 0 → 0 (match MATLAB intent)
    npmi[~valid_marg] = 0.0

    # Compute NPMI where joint > 0
    if np.any(valid_joint):
        lj = np.log(pxy[valid_joint])
        li = np.log(pxi[valid_joint])
        ljj = np.log(pxj[valid_joint])
        denom = -lj  # -log(PXY)
        # Guard tiny denom just in case
        denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
        npmi[valid_joint] = (lj - li - ljj) / denom

    # Average over all pairs (S choose 2), same denominator as MATLAB:
    n_pairs = (S * S - S) // 2
    # Note: positions not in lower triangle don't contribute (we only built npmi over those)
    return float(npmi.sum() / n_pairs)


def div_all_isolates_numpy(SSPrev_obs) -> float:
    """
    Overall strain diversity across the entire study period.
    Steps:
      1) Collapse time: totals per strain = sum over columns (time).
      2) Reciprocal Simpson: D = N*(N-1) / sum_i n_i*(n_i-1)
         (returns 0 if no isolates; equals richness when all n_i ∈ {0,1})
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    if X.size == 0:
        return 0.0
    totals = X.sum(axis=1)  # per-strain totals across time
    N = totals.sum()
    if N <= 1:
        return 0.0
    denom = np.sum(totals * (totals - 1.0))
    if denom <= 0:
        # all totals are 0 or 1 → diversity equals number of nonzero strains
        return float((totals > 0).sum())
    return float(N * (N - 1.0) / denom)

# as an alternative for the matlab version
def avg_prev_numpy(SSPrev_obs) -> float:
    """
    Average prevalence over time.
    Accepts array-like (T,) or (T,1). NaNs ignored.
    """

    Prevalence_obs = SSPrev_obs.sum(axis=0)
    # print("Prevalence_obs:", Prevalence_obs)
    y = np.asarray(Prevalence_obs, dtype=float).ravel()
    return float(np.nanmean(y)) if y.size else float("nan")


def var_prev_numpy(SSPrev_obs, ddof: int = 1) -> float:
    """
    Variance of prevalence over time (ignores NaNs).
    Prevalence_obs: array-like (T,) or (T,1)
    ddof: 1 for sample variance (MATLAB-like), 0 for population variance.
    """

    Prevalence_obs = SSPrev_obs.sum(axis=0)
    # print("Prevalence_obs:", Prevalence_obs)
    y = np.asarray(Prevalence_obs, dtype=float).ravel()
    if y.size == 0:
        return float("nan")
    # need at least 2 valid points for sample variance
    if ddof == 1 and np.sum(~np.isnan(y)) < 2:
        return float("nan")
    return float(np.nanvar(y, ddof=ddof))