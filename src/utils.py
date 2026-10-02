import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, spearmanr
from src import uq_Methods
import math
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
skript_ordner = Path(__file__).parent
skript_ordner = skript_ordner.parent
ziel_ordner = skript_ordner / "results"

class QuantileTabPFNAdapter(BaseEstimator, RegressorMixin):
    """Used for CQR, because it needs the models as an input for the needed quantile.
    TabPFN model has to be prefitted.
    """
    def __init__(self, model, alpha=0.5):
        self.model = model
        self.alpha = alpha

    def fit(self, X, y):
        # Das Basis-Modell ist bereits gefittet (cv="prefit")
        return self

    def predict(self, X):
        # Ruft TabPFN/TabICL mit dem spezifischen Quantil auf
        # Gibt ein 1D-Array für das eine angeforderte Quantil zurück
        preds = self.model.predict(X, output_type="quantiles", quantiles=[self.alpha])
        if isinstance(preds, list):
            return preds[0]
        elif preds.ndim == 2:
            return preds[:, 0]
        return preds

def mse_plot_multi_sets(y_true_list, y_pred_list, uncertainties_list, dataset_names=None, sigma=100, cols=3, title="MSE vs. Uncertainty across Datasets"):
    """Plots smoothed MSE curves against uncertainty-sorted samples for multiple individual datasets in a multi-panel subplot grid.
    Used to plot all given datasets for a complete overview of MSE vs. Uncertainty plots next to one another.

    Parameters
    ----------
    y_true_list : list of array-like
        List of 1D arrays containing ground-truth target values for each dataset.
    y_pred_list : list of array-like
        List of 1D arrays containing predicted target values for each dataset.
    uncertainties_list : list of array-like
        List of 1D arrays containing estimated uncertainties for each dataset.
    dataset_names : list of str, optional
        Names or labels corresponding to each entry in the dataset lists. If None, default labels 
        ("Dataset 1", "Dataset 2", ...) are generated.
    sigma : int, default=100
        Base standard deviation for Gaussian kernel smoothing applied to the sorted MSE values. 
        Scaled dynamically based on dataset size N.
    cols : int, default=3
        Number of columns in the subplot grid layout.
    title : str, default="MSE vs. Uncertainty across Datasets"
        Main figure title and file naming descriptor for the saved output plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object containing the rendered subplots grid.
    """
    num_datasets = len(y_true_list)
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(num_datasets)]

    rows = math.ceil(num_datasets / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True)
    
    # Flatten axes array for easy iteration
    axes_flat = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_datasets)))

    for i, (y_true, y_pred, unc, name) in enumerate(zip(y_true_list, y_pred_list, uncertainties_list, dataset_names)):
        ax = axes_flat[i]
        
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        u = np.asarray(unc).ravel()

        N = len(y_t)
        if N == 0:
            continue

        mse_raw = (y_p - y_t) ** 2
        correlation, p_value = spearmanr(u, mse_raw)

        # Sort by uncertainties
        sort_idx = np.argsort(u)
        mse_sorted = mse_raw[sort_idx]

        # Gaußian smoothing funcation from the paper
        effective_sigma = max(1, int(sigma * (N / 1000.0)))
        mse_smooth = gaussian_filter1d(mse_sorted, sigma=effective_sigma)
        mse_smooth = np.nan_to_num(mse_smooth)

        x_percentiles = np.linspace(0, 100, N)

        ax.plot(x_percentiles, mse_smooth, color=colors[i % len(colors)], linewidth=2.0)
        
        # Subplot Styling
        ax.set_title(f"{name}\n(Spearman $\\rho$: {correlation:.2f})", fontsize=11, fontweight='bold')
        ax.set_ylabel("MSE", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 100)

        print(f"[{name}] N={N} | Spearman-Korrelation: {correlation:.3f} (p: {p_value:.3e})")

    # Hide empty axes if the number of data sets does not form a perfect rectangle.
    for j in range(num_datasets, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Common x-label for bottom row
    fig.supxlabel(
        "lower uncertainty ← Samples sorted by uncertainty (%) → higher uncertainty",
        fontsize=11,
        y=0.01
    )
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    ziel_datei = ziel_ordner / "rq2" / f"rq2_MSE_{title}.png"
    os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close("all")

    return fig

def scatter_plot_coverage(y_true, medians, intervals, n_samples=30, title="Interval Coverage Scatter Plot"):
    """
    Were not used for the thesis, but they give a good overview for prediction behaviour.
    Improved Interval Scatter Plot:
    - Plots predicted medians vs true values with horizontal prediction intervals.
    - Color-codes covered intervals in GREEN and missed intervals in RED.
    """
    y_true = np.asarray(y_true).ravel()
    medians = np.asarray(medians).ravel()
    intervals = np.asarray(intervals)

    # Subsample indices for clean visual density
    n_pts = min(n_samples, len(y_true))
    indices = np.random.choice(len(y_true), size=n_pts, replace=False)

    y_sub = y_true[indices]
    med_sub = medians[indices]
    x_min = intervals[indices, 0]
    x_max = intervals[indices, 1]

    # Boolean mask: True if y_true is inside [x_min, x_max]
    inside_mask = (y_sub >= x_min) & (y_sub <= x_max)

    # Color mapping
    line_colors = np.where(inside_mask, "#2ca02c", "#d62728")  # Green / Red
    point_colors = np.where(inside_mask, "#1b681b", "#8c1111")

    fig, ax = plt.subplots(figsize=(8, 7))

    # Horizontal error bars (prediction intervals)
    ax.hlines(y_sub, x_min, x_max, colors=line_colors, alpha=0.8, linewidth=2, label="Interval")

    # Point estimates (Medians)
    ax.scatter(med_sub, y_sub, color=point_colors, zorder=3, s=40, label="Point Estimate (Median)")

    # Perfect prediction reference line (45 degree)
    min_val = min(y_sub.min(), med_sub.min(), x_min.min())
    max_val = max(y_sub.max(), med_sub.max(), x_max.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, label="Ideal (y_true = y_pred)")

    emp_cov = np.mean(inside_mask) * 100
    ax.set_title(f"{title}\nEmpirical Sample Coverage: {emp_cov:.1f}% ({np.sum(inside_mask)}/{n_pts})", fontsize=12)
    ax.set_xlabel("Predicted Value / Median", fontsize=11)
    ax.set_ylabel("True Target (y_true)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    ziel_datei = ziel_ordner / "rq4" / "1" / f"rq4_1_scatter_{title}.png"
    os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close("all")

def coverage_plot(dataset_len ,packs ,cols=3, title="Coverage Plot"):
    """Plots prediction intervals and test coverage across multiple datasets in a multi-panel subplot grid.
    Used for the 3 synthetic datasets, to plot the differences of MAPIE and PPD predictions visually.

    Parameters
    ----------
    dataset_len : int
        Total number of datasets or subplots to render.
    packs : tuple of iterables
        Unpacked collection containing data elements for each subplot in the following order:
        (X_test, X_train, y_train, y_pred, lower_bounds, upper_bounds, tab_lower_bounds, tab_upper_bounds, conf_level, ax).
    cols : int, default=3
        Number of columns in the subplot grid layout.
    title : str, default="Coverage Plot"
        Title descriptor used when constructing the saved plot output filename.

    Returns
    -------
    None
        The plot is rendered, saved to disk, and the figure is closed.
    """
    rows = math.ceil(dataset_len / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15,12), sharex=True)
    # Flatten axes array for easy iteration
    axes_flat = np.atleast_1d(axes).ravel()

    for (X_test, X_train, y_train, y_pred, lower_bounds, upper_bounds, tab_lower_bounds, tab_upper_bounds, conf_level, ax) in zip(*packs, axes_flat):     
        # Ploting training data as points
        ax.scatter(X_train, y_train, color="red", alpha=0.7, label=f"Trainingsdaten (N={len(X_train)})")
                
        # TabPFN prediciton
        ax.plot(X_test, y_pred, color="blue", label="TabPFN Vorhersage")
                
        # MAPIE uncertainty via shaded interval
        ax.fill_between(X_test.to_numpy().flatten(), lower_bounds, upper_bounds, color="blue", alpha=0.15, label=f"MAPIE {conf_level*100}% Intervall")
        ax.fill_between(X_test.to_numpy().flatten(), tab_lower_bounds, tab_upper_bounds, color="green", alpha=0.15, label=f"TabPFN {conf_level*100}% Intervall")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    for j in range(dataset_len, len(axes_flat)):
            fig.delaxes(axes_flat[j])
    

    plt.tight_layout()
    plt.suptitle("Coverage MAPPIE vs. PPD", y=1.02, fontsize=16, fontweight='bold')
    ziel_datei = ziel_ordner / "rq4" / "1" / f"rq4_1_coverage_{title}.png"
    os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close("all")

def plot_mp_convergence(B_runs, K_runs, title="Martingale Posterior Convergence"):
    """
    Create two line plots for the convergence of B and K (with uncertainty spread).
    This method is not used in this thesis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convergence of B with K=50
    b_vals = []
    b_means = []
    b_stds = []

    for ppd, (b, k) in B_runs:
        # get uncertainties
        uq = uq_Methods.martingale_uq(ppd)
        b_vals.append(b)
        b_means.append(np.mean(uq))
        b_stds.append(np.std(uq))

    b_vals = np.array(b_vals)
    b_means = np.array(b_means)
    b_stds = np.array(b_stds)

    #Plot B-Convergance on first axis
    ax1.plot(b_vals, b_means, 'o-', color='#1f77b4', linewidth=2, label='Mean Uncertainty')
    ax1.fill_between(b_vals, b_means - b_stds, b_means + b_stds, color='#1f77b4', alpha=0.2, label='±1 Std')
    ax1.set_title("A: Convergence if resampled paths ($B$)", fontweight='bold')
    ax1.set_xlabel("Number of independent posterior draws ($B$)")
    ax1.set_ylabel("Estimated Uncertainty")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Convergence of K with B=50
    k_vals = []
    k_means = []
    k_stds = []

    for ppd, (b, k) in K_runs:
        #get uncertainties
        uq = uq_Methods.martingale_uq(ppd)
        k_vals.append(k)
        k_means.append(np.mean(uq))
        k_stds.append(np.std(uq))

    k_vals = np.array(k_vals)
    k_means = np.array(k_means)
    k_stds = np.array(k_stds)

    #Plot K-Convergance on second axis
    ax2.plot(k_vals, k_means, 's-', color='#ff7f0e', linewidth=2, label='Mean Uncertainty')
    ax2.fill_between(k_vals, k_means - k_stds, k_means + k_stds, color='#ff7f0e', alpha=0.2, label='±1 Std')
    ax2.set_title("B: Convergence of forward steps beyond the training burn-in ($K$)", fontweight='bold')
    ax2.set_xlabel("Amound of forward samples ($K$)")
    ax2.set_ylabel("Estimated Uncertainty")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def mse_plot_multi_folds(y_true_list, y_pred_list, uncertainties_list, dataset_names=None, sigma=20, title="MSE vs. Uncertainty (Folds)", ax=None):
    """Plots smoothed Mean Squared Error (MSE) against samples sorted by estimated uncertainty across multiple datasets or cross-validation folds.

    Parameters
    ----------
    y_true_list : list of array-like
        List of 1D arrays containing ground-truth target values for each dataset or fold.
    y_pred_list : list of array-like
        List of 1D arrays containing predicted target values for each dataset or fold.
    uncertainties_list : list of array-like
        List of 1D arrays containing estimated uncertainties for each dataset or fold.
    dataset_names : list of str, optional
        Names or labels corresponding to each entry in the dataset lists. If None, default labels 
        ("Dataset 1", "Dataset 2", ...) are generated.
    sigma : int, default=20
        Base standard deviation for Gaussian kernel smoothing applied to the sorted MSE values. 
        Scaled dynamically based on the dataset size N.
    title : str, default="MSE vs. Uncertainty (Folds)"
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib Axes object on which to render the plot. If None, a new Figure and Axes 
        are created, and the plot is saved automatically to disk.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object containing the rendered plot.
    mean_rho : float
        Mean Spearman rank correlation coefficient across all processed datasets/folds.
    std_rho : float
        Standard deviation of the Spearman rank correlation coefficients across all processed datasets/folds.
    """
    num_datasets = len(y_true_list)
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(num_datasets)]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_datasets)))

    # Lists to gather data for mean plot
    grid_percentiles = np.linspace(0, 100, 500)  # X'es (500 Points)
    all_mse_interpolated = []                    # Gesammelte MSE-Curves
    spearman_rhos = []                          # Gesammelte Spearman-Coefficients

    # loop over all datasets
    for i, (y_true, y_pred, unc, name) in enumerate(zip(y_true_list, y_pred_list, uncertainties_list, dataset_names)):
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        u = np.asarray(unc).ravel()

        N = len(y_t)
        if N == 0:
            continue

        # Pointwise MSE & Spearman-Corelation
        mse_raw = (y_p - y_t) ** 2
        correlation, p_value = spearmanr(u, mse_raw)
        spearman_rhos.append(correlation)

        # Sort by uncertainty
        sort_idx = np.argsort(u)
        mse_sorted = mse_raw[sort_idx]
        #print(mse_sorted, sep='\n')
        #print(u, sep='\n')
 
        # Gaußian smoothing used in the paper
        effective_sigma = max(1, int(sigma * (N / 1000.0))) 
        mse_smooth = gaussian_filter1d(mse_sorted, sigma=effective_sigma)
        mse_smooth = np.nan_to_num(mse_smooth)

        # Original x-axis of specific dataset
        x_percentiles = np.linspace(0, 100, N)

        # Interpolate on common grid
        mse_interp = np.interp(grid_percentiles, x_percentiles, mse_smooth)
        all_mse_interpolated.append(mse_interp)

        label_text = f"{name} (Spearman $\\rho$: {correlation:.2f})"
        ax.plot(x_percentiles, mse_smooth, color=colors[i], linewidth=1.5, alpha=0.45, label=label_text)

        print(f"[{name}] N={N} | Spearman-Korrelation: {correlation:.3f} (p: {p_value:.3e})")

    # CALCS OF MEAN & STD
    if all_mse_interpolated:
        all_mse_array = np.array(all_mse_interpolated)  # Shape: (num_datasets, 500)
        
        # Mean and std of every point
        mean_mse = np.mean(all_mse_array, axis=0)
        std_mse = np.std(all_mse_array, axis=0)

        # Mean and std of spearman
        mean_rho = np.mean(spearman_rhos)
        std_rho = np.std(spearman_rhos)

        # MEAN-PLOT & STD-BAND 
        # Std with shading
        ax.fill_between(
            grid_percentiles, 
            np.maximum(0, mean_mse - std_mse),  # Verhindert negative MSE-Werte im Band
            mean_mse + std_mse, 
            color='black', 
            alpha=0.15, 
            label=r'Mean $\pm$ 1 Std'
        )

        # Mean plot extra thick
        ax.plot(
            grid_percentiles, 
            mean_mse, 
            color='black', 
            linewidth=3, 
            linestyle='-', 
            label=f"Mean (Spearman $\\rho$: {mean_rho:.2f} $\\pm$ {std_rho:.2f})"
        )

        print(f"\n[GESAMT] Mittlerer Spearman-Koeffizient: {mean_rho:.3f} ± {std_rho:.3f}")

    # Styling and Labels
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('MSE', fontsize=11)
    
    ax.set_xlabel(
        r"lower uncertainty $\leftarrow$ \textbf{Samples sorted by uncertainty (\%)} $\rightarrow$ higher uncertainty" 
        if plt.rcParams.get('text.usetex') else 
        "lower uncertainty ← Samples sorted by uncertainty (%) → higher uncertainty",
        fontsize=10,
        labelpad=8
    )

    ax.set_xlim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.legend(bbox_to_anchor=(0.91, 1), loc='upper left', borderaxespad=0, framealpha=0.6, fontsize=11)

    if created_fig:
        ziel_datei = ziel_ordner / "rq2" / f"rq2_multi_folds_{title}.png"
        os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
        plt.savefig(ziel_datei, dpi=300, bbox_inches="tight")
        plt.close()

    return ax, mean_rho, std_rho

def plot_sparsification_multi_sets(
    y_true_list, 
    y_pred_list, 
    uncertainties_list, 
    dataset_names=None, 
    show_mean=True, 
    show_oracle_mean=False,
    sigma=1.5,
    normalize=True,  # <-- NEU: Normalisiert die Fehler auf [0, 1] bezüglich des Initial-MSE
    title="MSE Sparsification Plot", 
    ax=None
):
    """Creates an MSE Sparsification Plot comparing model uncertainty estimations against an oracle across multiple datasets or cross-validation folds.

    Parameters
    ----------
    y_true_list : list of array-like
        List of 1D arrays containing ground-truth target values for each dataset or fold.
    y_pred_list : list of array-like
        List of 1D arrays containing predicted target values for each dataset or fold.
    uncertainties_list : list of array-like
        List of 1D arrays containing estimated uncertainties for each dataset or fold.
    dataset_names : list of str, optional
        Names or labels corresponding to each entry in the dataset lists. If None, default labels 
        ("Dataset 1", "Dataset 2", ...) are generated.
    show_mean : bool, default=True
        Whether to calculate and display the mean sparsification curve and standard deviation band 
        across all datasets/folds.
    show_oracle_mean : bool, default=False
        Whether to calculate and display the overall mean oracle curve.
    sigma : float, default=1.5
        Standard deviation for Gaussian kernel smoothing applied to the curves. Set to 0 to disable smoothing.
    normalize : bool, default=True
        If True, normalizes each curve by its initial MSE (at 0% samples removed) to allow 
        comparison across datasets with different target scales.
    title : str, default="MSE Sparsification Plot"
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Pre-existing Matplotlib Axes object on which to render the plot. If None, a new Figure and Axes 
        are created, and the plot is saved automatically to disk.

    Returns
    -------
    results : dict
        Dictionary containing metric summaries:
        
        - ``"ause_list"``: list of float
            AUSE (Area Under the Sparsification Error) scores for each dataset.
        - ``"auc_model_list"``: list of float
            AUC values for the model sparsification curves.
        - ``"auc_oracle_list"``: list of float
            AUC values for the optimal oracle sparsification curves.
        - ``"ause_mean"``: float
            Mean AUSE across all processed datasets/folds.
        - ``"ause_std"``: float
            Standard deviation of AUSE across all processed datasets/folds.
        - ``"auc_model_mean"``: float
            Mean model AUC across all processed datasets/folds.
        - ``"auc_model_std"``: float
            Standard deviation of model AUC across all processed datasets/folds.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object containing the rendered plot.
    """
    num_datasets = len(y_true_list)
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(num_datasets)]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_datasets)))

    grid_fractions = np.linspace(0, 1, 500)
    grid_percentiles = grid_fractions * 100

    all_sparsification_interpolated = []
    all_oracle_interpolated = []

    auc_model_list = []
    auc_oracle_list = []
    ause_list = []

    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    def apply_smoothing(y):
        if sigma > 0:
            return gaussian_filter1d(y, sigma=sigma, mode='nearest')
        return y

    # Loop through all datasets / folds
    for i, (y_true, y_pred, unc, name) in enumerate(zip(y_true_list, y_pred_list, uncertainties_list, dataset_names)):
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        u = np.asarray(unc).ravel()

        N = len(y_t)
        if N == 0:
            continue

        #Pointwise MSE
        squared_errors = (y_p - y_t) ** 2
        fractions_removed = np.linspace(0, 1, N)

        # MODEL SPARSIFICATION
        sort_unc_idx = np.argsort(u)
        sorted_errors_unc = squared_errors[sort_unc_idx]
        cum_mse_unc = np.cumsum(sorted_errors_unc) / np.arange(1, N + 1)
        sparsification_curve = cum_mse_unc[::-1]

        # ORACLE SPARSIFICATION
        sort_oracle_idx = np.argsort(squared_errors)
        sorted_errors_oracle = squared_errors[sort_oracle_idx]
        cum_mse_oracle = np.cumsum(sorted_errors_oracle) / np.arange(1, N + 1)
        oracle_curve = cum_mse_oracle[::-1]

        # Normalize, because different datasets have different spans of error (When getting different datasets as input instead of 5 Folds)
        initial_mse = sparsification_curve[0]
        if normalize and initial_mse > 0:
            sparsification_curve = sparsification_curve / initial_mse
            oracle_curve = oracle_curve / initial_mse

        # Interpolation onto the shared grid
        curve_interp = np.interp(grid_fractions, fractions_removed, sparsification_curve)
        oracle_interp = np.interp(grid_fractions, fractions_removed, oracle_curve)

        all_sparsification_interpolated.append(curve_interp)
        all_oracle_interpolated.append(oracle_interp)

        # AUSE CALCULATION
        auc_model = trapz_func(curve_interp, grid_fractions)
        auc_oracle = trapz_func(oracle_interp, grid_fractions)
        ause = auc_model - auc_oracle

        auc_model_list.append(auc_model)
        auc_oracle_list.append(auc_oracle)
        ause_list.append(ause)

        # GAUSSIAN SMOOTHING (just used the same smoothing function as with the MSE plot)
        curve_plot = apply_smoothing(curve_interp)
        oracle_plot = apply_smoothing(oracle_interp)

        alpha_val = 0.45 if show_mean else 0.85
        current_color = colors[i]

        label_text = f"{name} (AUSE: {ause:.3f})"
        ax.plot(grid_percentiles, curve_plot, color=current_color, linewidth=1.5, alpha=alpha_val, label=label_text)

        oracle_label_text = f"Oracle {name}"
        ax.plot(grid_percentiles, oracle_plot, color=current_color, linewidth=1.5, linestyle='--', alpha=alpha_val, label=oracle_label_text)

    # 2. Compute Mean & Std
    results = {
        'ause_list': ause_list,
        'auc_model_list': auc_model_list,
        'auc_oracle_list': auc_oracle_list,
        'ause_mean': np.mean(ause_list) if ause_list else np.nan,
        'ause_std': np.std(ause_list) if ause_list else np.nan,
        'auc_model_mean': np.mean(auc_model_list) if auc_model_list else np.nan,
        'auc_model_std': np.std(auc_model_list) if auc_model_list else np.nan,
    }

    #Get mean +/- std of all plots (Used for 5 Fold Cross Validation)
    if show_mean and all_sparsification_interpolated:
        all_curves_array = np.array(all_sparsification_interpolated)
        mean_curve = np.mean(all_curves_array, axis=0)
        std_curve = np.std(all_curves_array, axis=0)

        mean_curve_plot = apply_smoothing(mean_curve)
        std_curve_plot = apply_smoothing(std_curve)

        ax.fill_between(
            grid_percentiles, 
            np.maximum(0, mean_curve_plot - std_curve_plot), 
            mean_curve_plot + std_curve_plot, 
            color='black', 
            alpha=0.15, 
            label=r'Mean $\pm$ 1 Std'
        )

        mean_label = f"Mean Sparsification (AUSE: {results['ause_mean']:.3f} $\\pm$ {results['ause_std']:.3f})"
        ax.plot(grid_percentiles, mean_curve_plot, color='black', linewidth=3, linestyle='-', label=mean_label)

        if show_oracle_mean and all_oracle_interpolated:
            mean_oracle_curve = np.mean(np.array(all_oracle_interpolated), axis=0)
            mean_oracle_plot = apply_smoothing(mean_oracle_curve)
            ax.plot(grid_percentiles, mean_oracle_plot, color='red', linewidth=2, linestyle=':', label='Mean Oracle (Optimal UQ)')

    # 3. Styling & Labels
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Fraction of removed most uncertain samples (%)", fontsize=11, labelpad=8)
    
    # Adjust label on different settings
    if normalize:
        ax.set_ylabel("Normalized Remaining MSE (Relative to initial)", fontsize=11)
    else:
        ax.set_ylabel("Remaining Overall MSE", fontsize=11)

    ax.set_xlim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.legend(bbox_to_anchor=(0.91, 1), loc='upper left', borderaxespad=0, framealpha=0.6, fontsize=11)

    if created_fig:
        ziel_datei = ziel_ordner / "rq2" / f"rq2_spars_multi_{title}.png"
        os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
        plt.savefig(ziel_datei, dpi=300, bbox_inches="tight")
        plt.close()
    print("=======================================================================")
    print(results)
    print("=======================================================================")

    return results, ax
