from __future__ import annotations
import numpy as np
import pyvinecopulib as pv

DEFAULT_B_POSTSAMPLES = 50
DEFAULT_T_FWDSAMPLES = 500
DEFAULT_RHO = 0.99
FONG_ALPHA_C = 2.0
FONG_ALPHA_BETA = 1.0

#Old method not really used for RQ's
def martingale_uq(posterior):
    """Calculate Variance with the martingale output for every point."""
    #temp_pos = list(zip(*posterior))
    #return [np.var(temp_pos[i]) for i in range(0,len(temp_pos))]
    variances = []
    
    for punkt_daten in posterior:
        # punkt_daten["ysim"] hat die Shape (T_total, B_postsamples)
        # Wir wollen den letzten Zeitschritt [-1, :] über alle B_postsamples
        finale_vorhersagen = punkt_daten["ysim"][-1, :]
        
        # Varianz über die Stichproben berechnen
        punkt_varianz = np.var(finale_vorhersagen)
        variances.append(punkt_varianz)
        
    return variances

def total_martingale_uq(posterior, n_train, y_grid, alpha=0.1):
    """Calculate total uncertainty per point (epistemic + aleatoric)
    Computes total predictive uncertainty, epistemic/aleatoric variance components,
    and median intervals across all evaluation points from `get_posterior` output.

    Parameters
    ----------
    posterior_list : list[dict]
        Output list from `get_posterior`.
    y_grid : (K,) or (K, 1) array
        Support grid passed to `get_posterior`.
    n_train : int
        Number of training points (burn-in length).
    lo, hi : float, optional
        Percentile bounds for the interval (default: 5.0 and 95.0 for 90% coverage).

    Returns
    -------
    dict of NumPy arrays (shape: n_eval except median_intervals which is (n_eval, 2)):
        - "total_uncertainty_var": Total variance Var(Y_sim)
        - "total_uncertainty_std": Total standard deviation Std(Y_sim)
        - "epistemic_uncertainty_var": Epistemic variance of path means
        - "aleatoric_uncertainty_var": Mean aleatoric variance within paths
        - "median_intervals": (n_eval, 2) array of [lower, upper] interval bounds
        - "interval_width": Width of the prediction interval
    """
    # 1. Compute median intervals across posterior draws
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    intervals = median_intervals_from_resamples(posterior, y_grid, lo=lo, hi=hi)
    
    n_eval = len(posterior)
    total_var = np.empty(n_eval)
    epistemic_var = np.empty(n_eval)
    aleatoric_var = np.empty(n_eval)

    # 2. Extract variances from forward simulations
    for j, entry in enumerate(posterior):
        # ysim_fwd shape: (T_fwdsamples, B_postsamples)
        ysim_fwd = entry["ysim"][n_train:, :]
        
        # Total variance over all forward steps and draws
        total_var[j] = np.var(ysim_fwd, ddof=1)
        
        # Variance decomposition (Law of Total Variance)
        epistemic_var[j] = np.var(np.mean(ysim_fwd, axis=0), ddof=1)
        aleatoric_var[j] = np.mean(np.var(ysim_fwd, axis=0, ddof=1))

    return {
        "total_uncertainty_var": epistemic_var+aleatoric_var,
        "total_uncertainty_std": np.sqrt(total_var),
        "epistemic_uncertainty_var": epistemic_var,
        "aleatoric_uncertainty_var": aleatoric_var,
        "median_intervals": intervals,
        "interval_width": intervals[:, 1] - intervals[:, 0],
    }












# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def get_posterior(
    y_grid: np.ndarray,
    n_train: int,
    B_postsamples: int,
    T_fwdsamples: int,
    cdf_conditionals: np.ndarray,
    rho: float = DEFAULT_RHO,
    seed: int = 100,
    alpha_schedule=None,
    alpha_dimension: int | None = None,
    n_eff=None,
    shared_uniform: bool = True,
    use_blowup: bool = True,
) -> list[dict]:
    """
    Run the martingale posterior update and return posterior samples.

    Parameters
    ----------
    y_grid : (K,) or (K, 1) array
        y-values at which the conditional CDF is evaluated (the support grid).
    n_train : int
        Number of observed training points (burn-in length).
    B_postsamples : int
        Number of independent posterior draws (repetitions).
    T_fwdsamples : int
        Number of forward steps beyond the training burn-in.
    cdf_conditionals : (K, n_eval) array
        Pre-computed conditional CDF evaluations of the predictor on y_grid
        for each of the n_eval test points.
    rho : float
        Gaussian copula serial correlation parameter.
    seed : int
        Base RNG seed.
    alpha_schedule : callable(i) -> float, optional
        Learning-rate schedule. If omitted, callers must provide
        ``alpha_dimension`` so the dimension-adaptive default
        ``C * (i + 1)^(-beta)`` can be constructed with
        ``beta = 0.5 + 2 / (1.1 * d + 4.0)`` and ``C = 2 ** beta``.
        Use ``make_alpha_schedules`` to obtain the supported choices.
    alpha_dimension : int, optional
        Feature dimension ``d`` used by the default martingale schedule when
        ``alpha_schedule`` is omitted.
    n_eff : None, scalar, or (n_eval,) array, optional
        Effective sample size for each evaluation point. When provided,
        callers may use it inside ``alpha_schedule`` by defining a callable
        that accepts two arguments ``alpha_schedule(k, n_eff)`` where ``k``
        is the forward-step index starting at 0. Existing one-argument
        schedules ``alpha_schedule(i)`` continue to work unchanged.
    shared_uniform : bool
        If True, a single uniform is drawn per step and shared across all
        test points (faster, matches the original implementation).
    use_blowup : bool
        If True, scale alpha by the finite-horizon blowup factor
        ``blowup_integral(n_train, d, T_fwdsamples)`` when
        ``alpha_dimension`` is provided. If False, do not apply this scaling.

    Returns
    -------
    list of dicts, one per test point (length n_eval).
    Each dict has:
        ``"ysim"`` — (T_total, B_postsamples) array of simulated y values
        ``"cdf"``  — (K, B_postsamples) array of updated CDF vectors
    """
    cdf_conditionals = np.asarray(cdf_conditionals, dtype=float)
    K, n_eval = cdf_conditionals.shape
    support = np.asarray(y_grid).ravel()
    if n_eff is None:
        n_eff_arr = None
    else:
        n_eff_arr = np.asarray(n_eff, dtype=float)
        if n_eff_arr.ndim == 0:
            n_eff_arr = np.full(n_eval, float(n_eff_arr))
        elif n_eff_arr.shape != (n_eval,):
            raise ValueError(
                f"n_eff must be scalar or shape ({n_eval},), got {n_eff_arr.shape}"
            )

    if alpha_schedule is None:
        if alpha_dimension is None:
            raise ValueError(
                "alpha_dimension is required when alpha_schedule is omitted; "
                "get_posterior no longer assumes d=1."
            )
        if alpha_dimension <= 0:
            raise ValueError(f"alpha_dimension must be positive, got {alpha_dimension}")

    rho_arr = np.array([[float(rho)]], dtype="float64")
    cop = pv.Bicop(family=pv.BicopFamily.gaussian, rotation=0, parameters=rho_arr)
    eps = 1e-6

    N = T_fwdsamples + n_train
    ysim_all = np.zeros((n_eval, N, B_postsamples))
    cdf_all = np.zeros((n_eval, K, B_postsamples))

    if use_blowup and alpha_dimension is not None:
        alpha_blowup = blowup_integral(n_train, alpha_dimension, T_fwdsamples)
    else:
        alpha_blowup = 1.0

    for b in range(B_postsamples):
        np.random.seed(seed + b)
        cdf_b = np.clip(cdf_conditionals.T.copy(), eps, 1.0 - eps)  # (n_eval, K)
        ysim = np.zeros((n_eval, N))

        for i in range(n_train + 1, N):
            if alpha_schedule is None:
                alpha = default_alpha_schedule(i, alpha_dimension)
            else:
                alpha = alpha_schedule(i)
            alpha = np.asarray(alpha, dtype=float)
            if alpha.ndim == 0:
                alpha = np.full((n_eval, 1), float(alpha))
            elif alpha.shape == (n_eval,):
                alpha = alpha[:, None]
            elif alpha.shape != (n_eval, 1):
                raise ValueError(
                    "alpha_schedule must return a scalar, shape (n_eval,), or "
                    f"shape (n_eval, 1); got {alpha.shape}"
                )

            alpha = alpha * alpha_blowup
            if shared_uniform:
                u_val = np.random.uniform()
                u2 = np.full((n_eval, K), u_val)
            else:
                u_vals = np.random.uniform(size=n_eval)
                u2 = np.repeat(u_vals[:, None], K, axis=1)

            u = np.stack([np.clip(cdf_b, eps, 1.0 - eps), np.clip(u2, eps, 1.0 - eps)], axis=-1).reshape(-1, 2)
            h = cop.hfunc2(u).reshape(n_eval, K)
            cdf_b = (1 - alpha) * cdf_b + alpha * h
            cdf_b /= np.maximum(np.max(cdf_b, axis=1, keepdims=True), eps)
            cdf_b = np.clip(cdf_b, eps, 1.0 - eps)

            if shared_uniform:
                idx = np.array([np.searchsorted(cdf_b[j], u_val) for j in range(n_eval)])
                idx = np.clip(idx, 0, K - 1)[:, None]
            else:
                idx = np.searchsorted(cdf_b, u_vals[:, None])
                idx = np.clip(idx, 0, K - 1)

            ysim[:, i] = support[idx].squeeze()

        ysim_all[:, :, b] = ysim
        cdf_all[:, :, b] = cdf_b

    return [{"ysim": ysim_all[j], "cdf": cdf_all[j]} for j in range(n_eval)]


# ---------------------------------------------------------------------------
# Alpha schedules
# ---------------------------------------------------------------------------

def blowup_integral(n, d, T):
    gamma = 4.0 / (1.1 * d + 4.0)
    return 1.0 / np.sqrt(1.0 - (1.0 + T / n) ** (-gamma))

def _build_alpha_schedule(C: float, beta: float):
    return lambda i: C * (i + 1) ** (-beta)

def make_alpha_schedules(d: int) -> dict:
    """
    Return the supported alpha schedules for feature dimension *d*.

    Schedules
    ---------
    default : ``beta = 0.5 + 2 / (1.1 * d + 4.0)``, ``C = 2 ** beta``
    fong    : ``C = 2``, ``beta = 1``

    Returns
    -------
    dict mapping name -> {"C": float, "beta": float, "fn": callable(i) -> float}
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")

    beta_default = 0.5 + 2 / (1.1 * d + 4.0)

    params = {
        "default": (2 ** beta_default, beta_default),
        "fong": (FONG_ALPHA_C, FONG_ALPHA_BETA),
    }

    return {
        name: {"C": C, "beta": beta, "fn": _build_alpha_schedule(C, beta)}
        for name, (C, beta) in params.items()
    }


def default_alpha_schedule(i: int, d: int) -> float:
    """
    Return the dimension-adaptive default martingale learning rate.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    beta = 0.5 + 2 / (1.1 * d + 4.0)
    C = 2 ** beta
    return C * (i + 1) ** (-beta)


def fong_alpha_schedule(i: int) -> float:
    """Return the Fong et al. learning rate ``2 * (i + 1)^(-1)``."""
    return FONG_ALPHA_C * (i + 1) ** (-FONG_ALPHA_BETA)


# ---------------------------------------------------------------------------
# Interval extraction
# ---------------------------------------------------------------------------

def median_intervals_from_resamples(
    res_list: list[dict],
    y_grid: np.ndarray,
    lo: float = 5.0,
    hi: float = 95.0,
) -> np.ndarray:
    """
    Extract credible intervals on the posterior median from ``get_posterior``
    output.

    For each test point the function interpolates the median from each
    posterior CDF draw, then returns the [lo, hi] percentile bounds across
    draws.

    Parameters
    ----------
    res_list : list of dicts — output of ``get_posterior``
    y_grid   : (K,) or (K, 1) array — same grid passed to ``get_posterior``
    lo, hi   : float — percentile bounds (default: 5th and 95th)

    Returns
    -------
    intervals : (n_eval, 2) array  — columns are [lower, upper]
    """
    y_grid_1d = np.asarray(y_grid).ravel()
    out = []
    for entry in res_list:
        # entry["cdf"] shape: (K, B_postsamples)
        medians = [
            np.interp(0.5, entry["cdf"][:, b], y_grid_1d)
            for b in range(entry["cdf"].shape[1])
        ]
        out.append(np.percentile(medians, [lo, hi]))
    return np.stack(out)  # (n_eval, 2)
