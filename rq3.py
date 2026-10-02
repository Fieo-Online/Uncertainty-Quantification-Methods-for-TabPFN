from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from mapie.regression import ConformalizedQuantileRegressor
from pathlib import Path
from src import dataloader, uq_Methods, utils
from rq1 import evaluate_interval_quality
from rq2 import get_cdf

skript_ordner = Path(__file__).parent
ziel_ordner = skript_ordner / "results" / "rq3" 
os.environ["TABPFN_TOKEN"] = "Your PriorLabs TabPFN Token"

def run_rq3_experiment(n_sample_sizes=[25, 50, 75, 100], noise_level_list=[0.2, 0.5, 1.0, 1.5],noise_mode="growing_feature", scenario="clean"):
    """Executes the full experimental pipeline for Research Question 3 (RQ3).

    Evaluates the scaling and robustness behaviors of Martingale Posterior (MAP) 
    uncertainty quantification and MAPIE conformalized quantile regression across 
    varying training set sizes and injected label noise regimes.

    Parameters
    ----------
    n_sample_sizes : list of int, default=[25, 50, 75, 100]
        List of training sample sizes (N) to evaluate during dataset size scaling.
    noise_level_list : list of float, default=[0.2, 0.5, 1.0, 1.5]
        List of noise scaling factors applied to standard deviation of target values.
    noise_mode : {"growing_feature", "growing_target", "constant"}, default="growing_feature"
        Noise generation scheme used when calling ``inject_noise``.
    scenario : str, default="clean"
        Name of the experimental scenario passed to the data loader.

        - ``"clean"``:
            Generates three synthetic datasets without noise:
            Linear, Parabola, and Constant.
        
        - ``"tabarena"``:
            Loads selected real-world regression datasets from the
            TabArena benchmark via OpenML. Nominal features are encoded
            as integer values and all features and targets are converted
            to ``float32``.

    Returns
    -------
    None
        Appends metrics to ``rq3results.csv`` and saves scaling/noise evaluation plots.
    """
    print("==================================================")
    print(f" Running RQ3 Evaluation for : {scenario}")
    print("==================================================\n")
    
    datasets = dataloader.generate_scenario_data(scenario, 1000)
    
    for X, y, name in datasets:
        print(f"\n---> Evaluating Dataset: {name}")

        if len(X) > 500 and X.shape[1] > 15:
            X = X.sample(n=500)
            y = y.loc[X.index]
            print(f"{name} has to many samples and features so the training size is reduced to 200.")
        elif len(X) > 2000:
            X = X.sample(n=2000)
            y = y.loc[X.index]
            print(f"{name} has to many samples so the training size is reduced to 1000.")

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)
        y_std = np.std(y_train_full)

        # -------------------------------------------------------------
        # Part 1: Dataset Size Experiment
        # -------------------------------------------------------------
        print("Running Dataset Size Scaling Experiment...")
        metrics_list_MAP = []
        metrics_list_MAPPIE = []

        for N in n_sample_sizes:
            print(f"Size: {N}")
            for seed in range(5):
                # Ziehe korrekte Subsamples basierend auf N und Seed
                X_sample = X_train_full.sample(n=N)
                y_sample = y_train_full.loc[X_sample.index]

                # 1. Evaluierung Martingale Posterior (MAP)
                metrics_map = evaluate_uq_methods(X_sample, y_sample, X_test, y_test, UQ_Method="MAP")
                if metrics_map is not None:
                    metrics_list_MAP.append({
                        "dataset": name,
                        "N": N,
                        "seed": seed,
                        "MAP Epi UQ": np.mean(metrics_map["MAP Epi UQ"]),
                        "MAP Alea UQ": np.mean(metrics_map["MAP Alea UQ"]),
                        "MAP Total UQ": np.mean(metrics_map["MAP Total UQ"])
                    })

                # 2. Evaluierung MAPIE
                metrics_mapie = evaluate_uq_methods(X_sample, y_sample, X_test, y_test, UQ_Method="MAPPIE")
                if metrics_mapie is not None:
                    metrics_list_MAPPIE.append({
                        "dataset": name,
                        "N": N,
                        "seed": seed,
                        "MAPPIE Coverage": metrics_mapie["MAPPIE Coverage"],
                        "MAPPIE Mean Width": metrics_mapie["MAPPIE Mean Width"],
                    })

        df_results_MAP = pd.DataFrame(metrics_list_MAP)
        df_results_MAPPIE = pd.DataFrame(metrics_list_MAPPIE)
        df_results_MAP.to_csv('rq3results.csv', mode='a')
        df_results_MAPPIE.to_csv('rq3results.csv', mode='a')
        print(df_results_MAP.to_string(index=False))
        print(df_results_MAPPIE.to_string(index=False))
        
        # Plots für Teil 1 generieren
        if not df_results_MAP.empty:
            plot_rq3_dataset_size_MAP(df_results_MAP, dataset_name=name)
        if not df_results_MAPPIE.empty:
            plot_rq3_dataset_size_MAPPIE(df_results_MAPPIE, dataset_name=name)

        if len(noise_level_list) > 0:
            print(f"Running Noise Experiment ({noise_mode})...")
            noise_metrics_MAP = []
            noise_metrics_MAPPIE = []
            fixed_N = min(100, len(X_train_full))

        for noise_scale in noise_level_list:
            print(f"Noise Level: {noise_scale}")
            for seed in range(5):
                np.random.seed(seed)
                
                # Verwende die neue inject_noise Funktion
                y_train_noisy = inject_noise(
                    y=y_train_full, 
                    X=X_train_full, 
                    noise_scale=noise_scale, 
                    mode=noise_mode,
                    feature_idx=0
                )

                X_sample = X_train_full.sample(n=fixed_N)
                y_sample = y_train_noisy.loc[X_sample.index]

                # MAP Evaluierung
                metrics_map = evaluate_uq_methods(X_sample, y_sample, X_test, y_test, UQ_Method="MAP")
                if metrics_map is not None:
                    # Punktweise Korrelation zwischen geschätzter Unsicherheit und absolutem Fehler
                    spear_alea, _ = spearmanr(metrics_map["abs_errors"], metrics_map["MAP Alea UQ"])
                    spear_tot, _ = spearmanr(metrics_map["abs_errors"], metrics_map["MAP Total UQ"])

                    noise_metrics_MAP.append({
                        "dataset": name,
                        "noise_scale": noise_scale,
                        "seed": seed,
                        "noise_mode": noise_mode,
                        "MAP Epi UQ": np.mean(metrics_map["MAP Epi UQ"]),
                        "MAP Alea UQ": np.mean(metrics_map["MAP Alea UQ"]),
                        "MAP Total UQ": np.mean(metrics_map["MAP Total UQ"]),
                        "Spearman_Alea_Error": spear_alea,
                        "Spearman_Total_Error": spear_tot
                    })

                # MAPIE Evaluierung
                metrics_mapie = evaluate_uq_methods(X_sample, y_sample, X_test, y_test, UQ_Method="MAPPIE")
                if metrics_mapie is not None:
                    noise_metrics_MAPPIE.append({
                        "dataset": name,
                        "noise_scale": noise_scale,
                        "noise_mode": noise_mode,
                        "seed": seed,
                        "MAPPIE Coverage": metrics_mapie["MAPPIE Coverage"],
                        "MAPPIE Mean Width": metrics_mapie["MAPPIE Mean Width"]
                    })
        print("Creating Dataframes")
        df_noise_MAP = pd.DataFrame(noise_metrics_MAP)
        df_noise_MAPPIE = pd.DataFrame(noise_metrics_MAPPIE)
        print("Finished")
        print(df_noise_MAP.to_string(index=False))
        print(df_noise_MAPPIE.to_string(index=False))
        df_noise_MAP.to_csv('rq3results.csv', mode='a')
        df_noise_MAPPIE.to_csv('rq3results.csv', mode='a')
        if not df_noise_MAP.empty:
            plot_rq3_dataset_noise_MAP(df_noise_MAP, dataset_name=name)
        if not df_noise_MAPPIE.empty:
            plot_rq3_dataset_noise_MAPPIE(df_noise_MAPPIE, dataset_name=name)


def evaluate_uq_methods(X_train, y_train, X_test, y_test, UQ_Method="MAPPIE"):
    """Evaluates Uncertainty Quantification (UQ) performance using either MAPIE conformal prediction or Martingale Posterior decomposition.

    Parameters
    ----------
    X_train : pandas.DataFrame or array-like of shape (n_train_samples, n_features)
        Feature matrix for training data.
    y_train : pandas.Series or array-like of shape (n_train_samples,)
        Target values for training data.
    X_test : pandas.DataFrame or array-like of shape (n_test_samples, n_features)
        Feature matrix for test evaluation.
    y_test : pandas.Series or array-like of shape (n_test_samples,)
        Ground-truth target values for test evaluation.
    UQ_Method : {"MAPPIE", "MAP"}, default="MAPPIE"
        Uncertainty Quantification method to evaluate:

        - ``"MAPPIE"``: Conformalized Quantile Regression using MAPIE to compute coverage rate, interval width, and bounds.
        - ``"MAP"``: Martingale Posterior decomposition to compute total, epistemic, and aleatoric uncertainties along with prediction errors.

    Returns
    -------
    dict or None
        Dictionary containing method-specific metrics if evaluation succeeds, or ``None`` if sample requirements are not met or execution fails:

        If ``UQ_Method == "MAPPIE"``:
            - **"MAPPIE Coverage"** (*float*): Empirical coverage rate of the 90% prediction intervals.
            - **"MAPPIE Mean Width"** (*float*): Average width of the prediction intervals across test instances.
            - **"low"** (*numpy.ndarray* of shape `(n_test_samples,)`): Lower prediction interval bounds.
            - **"high"** (*numpy.ndarray* of shape `(n_test_samples,)`): Upper prediction interval bounds.

        If ``UQ_Method == "MAP"``:
            - **"MAP Total UQ"** (*numpy.ndarray* of shape `(n_test_samples,)`): Total uncertainty variance per test point.
            - **"MAP Epi UQ"** (*numpy.ndarray* of shape `(n_test_samples,)`): Epistemic uncertainty variance per test point.
            - **"MAP Alea UQ"** (*numpy.ndarray* of shape `(n_test_samples,)`): Aleatoric uncertainty variance per test point.
            - **"abs_errors"** (*numpy.ndarray* of shape `(n_test_samples,)`): Absolute prediction errors on the test set.
    """

    # ----------------------------------------------------------------------------------------
    # Testing MAPIE Coverage and Intervals
    # ----------------------------------------------------------------------------------------
    if UQ_Method == "MAPPIE":
        if len(X_train) < 37:
            print(f"Not enough samples. {len(X_train)} are not enough training samples, at least 37 training sample are needed for confidence level 0.9.")
            return None  # MAPIE benötigt genügend Punkte für Split-Conformal
            
        X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.42
        )
        regressor = TabPFNRegressor(ignore_pretraining_limits=True)
        #print("Fitting regressor")
        regressor.fit(X_train_fit, y_train_fit)
        
        try:
            #print("Starting CQR")
            model_low = utils.QuantileTabPFNAdapter(regressor, alpha=0.05)
            model_high = utils.QuantileTabPFNAdapter(regressor, alpha=0.95)
            model_mid = utils.QuantileTabPFNAdapter(regressor)
            mapie_model = ConformalizedQuantileRegressor(estimator=[model_low, model_high, model_mid], prefit=True, confidence_level=0.9)
            #print("Calibration starting")
            mapie_model.conformalize(X_cal, y_cal)
            #print("Calibration finished")
            #print("Prediction starting")
            y_pred, y_pis = mapie_model.predict_interval(X_test)
            #print("Prediction finished")
            mapie_low = y_pis[:, 0, 0]
            mapie_high = y_pis[:, 1, 0]

            metrics = evaluate_interval_quality(y_test, mapie_low, mapie_high, 0.1)
            return {
                "MAPPIE Coverage": metrics["coverage_rate"],
                "MAPPIE Mean Width": metrics["mean_width"],
                "low": mapie_low,
                "high": mapie_high
            }
        except Exception as e:
            print(f"MAPIE failed for sample size N={len(X_train)}: {e}")
            return None

    # ----------------------------------------------------------------------------------------
    # Testing MAP uncertainty decay
    # ----------------------------------------------------------------------------------------
    if UQ_Method == "MAP":
        regressor = TabPFNRegressor(ignore_pretraining_limits=True)
        #print("Fitting regressor")
        regressor.fit(X_train, y_train)
        
        # Punktschätzung (Mittelwert oder Median aus Quantilen)
        #print("Starting normal prediction")
        y_pred = regressor.predict(X_test)
        #print("Finished")
        abs_errors = np.abs(y_test - y_pred)
        
        quantiles_grid = np.linspace(0.01, 0.99, 99)
        #print("Starting quantile prediction")
        preds_quantiles = regressor.predict(X_test, output_type="quantiles", quantiles=quantiles_grid)
        #print("Finished")
        y_std_tr = np.std(y_train)
        y_min = y_train.min() - 3.0 * y_std_tr
        y_max = y_train.max() + 3.0 * y_std_tr
        y_grid = np.linspace(y_min, y_max, 99)

        cdf_conditionals = get_cdf(y_grid, preds_quantiles, quantiles_grid)
        #print("Getting Martingale Posteriors")
        post = uq_Methods.get_posterior(
            y_grid=y_grid,
            n_train=len(X_train),
            B_postsamples=50,
            T_fwdsamples=50,
            cdf_conditionals=cdf_conditionals,
            alpha_dimension=X_train.shape[1],
            use_blowup=True,
        )
        #print("Finished")
        uqs = uq_Methods.total_martingale_uq(post, len(X_train), y_grid)
        
        # Rückgabe der Arrays pro Testpunkt!
        return {
            "MAP Total UQ": uqs["total_uncertainty_var"],       # Array der Länge len(X_test)
            "MAP Epi UQ": uqs["epistemic_uncertainty_var"],   # Array der Länge len(X_test)
            "MAP Alea UQ": uqs["aleatoric_uncertainty_var"], # Array der Länge len(X_test)
            "abs_errors": abs_errors                            # Array der Länge len(X_test)
        }


# ============================================================================================
# PLOTTING FUNCTIONS
# ============================================================================================

def plot_rq3_dataset_size_MAP(df_results, dataset_name="", title=""):
    """Plots and saves Martingale Posterior variance decay (epistemic, aleatoric, and total) across increasing sample sizes.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing experimental results with columns ``"N"``, ``"MAP Epi UQ"``, 
        ``"MAP Alea UQ"``, and ``"MAP Total UQ"``.
    dataset_name : str, default=""
        Identifier or name of the dataset used for plot titling and file naming.
    title : str, default=""
        Additional title suffix appended to the output image filename.

    Returns
    -------
    None
        Generates a logarithmic scale line plot with error bands and saves the image to disk.
    """
    #Aggregate seed values and calculate mean and std
    df_agg = df_results.groupby("N").agg({
        "MAP Epi UQ": ["mean", "std"],
        "MAP Alea UQ": ["mean", "std"],
        "MAP Total UQ": ["mean", "std"]
    }).reset_index()

    N_vals = df_agg["N"]

    plt.figure(figsize=(8, 5))
    
    # Epistemische Unsicherheit
    ep_mean = df_agg[("MAP Epi UQ", "mean")]
    ep_std = df_agg[("MAP Epi UQ", "std")]
    plt.plot(N_vals, ep_mean, label="Epistemic Uncertainty", color="tab:blue", marker="o")
    plt.fill_between(N_vals, ep_mean - ep_std, ep_mean + ep_std, color="tab:blue", alpha=0.15)

    # Aleatorische Unsicherheit
    al_mean = df_agg[("MAP Alea UQ", "mean")]
    al_std = df_agg[("MAP Alea UQ", "std")]
    plt.plot(N_vals, al_mean, label="Aleatoric Uncertainty", color="tab:orange", marker="s")
    plt.fill_between(N_vals, al_mean - al_std, al_mean + al_std, color="tab:orange", alpha=0.15)

    # Gesamte Unsicherheit
    tot_mean = df_agg[("MAP Total UQ", "mean")]
    plt.plot(N_vals, tot_mean, label="Total Uncertainty", color="tab:green", linestyle="--", marker="^")

    plt.yscale("log")  # WICHTIG: Log-Skala, damit kleine Varianzunterschiede sichtbar sind!
    plt.xlabel("Sample Size (N)", fontsize=11)
    plt.ylabel("Predictive Variance (Log Scale)", fontsize=11)
    plt.title(f"Martingale Posterior: Uncertainty Decay vs. Dataset Size ({dataset_name})", fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.tight_layout()
    ziel_datei = ziel_ordner / f"rq3_map_variance_decay_{dataset_name}{title}.png"
    os.makedirs(os.path.dirname(ziel_ordner), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close()

def plot_rq3_dataset_size_MAPPIE(df_results, dataset_name="", title=""):
    """Plots and saves MAPIE conformal prediction coverage rate and average interval width scaling across sample sizes.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing experimental results with columns ``"N"``, ``"MAPPIE Coverage"``, 
        and ``"MAPPIE Mean Width"``.
    dataset_name : str, default=""
        Identifier or name of the dataset used for plot titling and file naming.
    title : str, default=""
        Additional title suffix appended to the output image filename.

    Returns
    -------
    None
        Generates a dual-axis line plot with standard deviation bands and saves the figure to disk.
    """
    df_agg = df_results.groupby("N").agg({
        "MAPPIE Coverage": ["mean", "std"],
        "MAPPIE Mean Width": ["mean", "std"]
    }).reset_index()

    N_vals = df_agg["N"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Sample Size (N)", fontsize=11)
    ax1.set_ylabel("MAPIE Coverage Rate", color=color, fontsize=11)
    cov_mean = df_agg[("MAPPIE Coverage", "mean")]
    cov_std = df_agg[("MAPPIE Coverage", "std")]
    ax1.plot(N_vals, cov_mean, color=color, marker="o", label="Coverage")
    ax1.fill_between(N_vals, cov_mean - cov_std, cov_mean + cov_std, color=color, alpha=0.15)
    ax1.axhline(0.90, color="black", linestyle=":", label="Target (90%)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:purple"
    ax2.set_ylabel("MAPIE Mean Interval Width", color=color, fontsize=11)
    width_mean = df_agg[("MAPPIE Mean Width", "mean")]
    width_std = df_agg[("MAPPIE Mean Width", "std")]
    ax2.plot(N_vals, width_mean, color=color, marker="s", linestyle="--", label="Interval Width")
    ax2.fill_between(N_vals, width_mean - width_std, width_mean + width_std, color=color, alpha=0.15)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"MAPIE Conformal Coverage & Width vs. Dataset Size ({dataset_name})", fontsize=12)
    fig.tight_layout()
    ziel_datei = ziel_ordner / f"rq3_mapie_scaling_{dataset_name}{title}.png"
    os.makedirs(os.path.dirname(ziel_ordner), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close()

def plot_rq3_dataset_noise_MAP(df_results, dataset_name="", title=""):
    """Plots and saves Martingale Posterior uncertainty decomposition response to synthetic label noise injection.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing experimental results with columns ``"noise_scale"``, ``"MAP Epi UQ"``, 
        ``"MAP Alea UQ"``, ``"MAP Total UQ"``, ``"Spearman_Alea_Error"``, and optionally ``"noise_mode"``.
    dataset_name : str, default=""
        Identifier or name of the dataset used for plot titling and file naming.
    title : str, default=""
        Additional title suffix appended to the output image filename.

    Returns
    -------
    None
        Generates a line plot showing uncertainty variance response to noise scale and saves the figure to disk.
    """
    df_agg = df_results.groupby("noise_scale").agg({
        "MAP Epi UQ": ["mean", "std"],
        "MAP Alea UQ": ["mean", "std"],
        "MAP Total UQ": ["mean", "std"],
        "Spearman_Alea_Error": ["mean"]
    }).reset_index()
    if "noise_mode" in df_results.columns:
        noise_mode_str = str(df_results["noise_mode"].iloc[0])
        print(noise_mode_str)
    else:
        noise_mode_str = "constant"
        print(noise_mode_str)
    noise_scales = df_agg["noise_scale"]
    alea_means = df_agg[("MAP Alea UQ", "mean")]
    epi_means = df_agg[("MAP Epi UQ", "mean")]
    mean_spearman = df_agg[("Spearman_Alea_Error", "mean")].mean()

    plt.figure(figsize=(8, 5))
    
    plt.plot(noise_scales, alea_means, label="Aleatoric Uncertainty", color="tab:orange", marker="s")
    plt.plot(noise_scales, epi_means, label="Epistemic Uncertainty", color="tab:blue", marker="o")
    plt.plot(noise_scales, df_agg[("MAP Total UQ", "mean")], label="Total Uncertainty", color="tab:green", linestyle="--", marker="^")

    plt.xlabel("Noise Scale (Multiplier of $\\sigma_y$)", fontsize=11)
    plt.ylabel("Mean Uncertainty Variance", fontsize=11)
    plt.title(f"Martingale Posterior under Label Noise ({dataset_name})\nAvg. Error-Uncertainty Correlation (Spearman $\\rho = {mean_spearman:.2f}$)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    ziel_datei = ziel_ordner / f"rq3_map_noise_injection_{dataset_name}{title}_{noise_mode_str}.png"
    os.makedirs(os.path.dirname(ziel_ordner), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close()

def plot_rq3_dataset_noise_MAPPIE(df_results, dataset_name="",title=""):
    """Plots and saves MAPIE conformal prediction interval width and coverage trends under varying levels of noise injection.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing experimental results with columns ``"noise_scale"``, ``"MAPPIE Coverage"``, 
        ``"MAPPIE Mean Width"``, and optionally ``"noise_mode"``.
    dataset_name : str, default=""
        Identifier or name of the dataset used for plot titling and file naming.
    title : str, default=""
        Additional title suffix appended to the output image filename.

    Returns
    -------
    None
        Generates a dual-axis line plot and saves the resulting figure as a PNG file to disk.
    """
    df_agg = df_results.groupby("noise_scale").agg({
        "MAPPIE Coverage": ["mean", "std"],
        "MAPPIE Mean Width": ["mean", "std"],
    }).reset_index()
    if "noise_mode" in df_results.columns:
        noise_mode_str = str(df_results["noise_mode"].iloc[0])
        print(noise_mode_str)
    else:
        noise_mode_str = "constant"
        print(noise_mode_str)
    noise_scales = df_agg["noise_scale"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Noise Scale (Multiplier of $\\sigma_y$)", fontsize=11)
    ax1.set_ylabel("MAPIE Coverage Rate", color=color, fontsize=11)
    cov_mean = df_agg[("MAPPIE Coverage", "mean")]
    ax1.plot(noise_scales, cov_mean, color=color, marker="o", label="Coverage")
    ax1.axhline(0.90, color="black", linestyle=":", label="Target (90%)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:purple"
    ax2.set_ylabel("MAPIE Mean Interval Width", color=color, fontsize=11)
    width_mean = df_agg[("MAPPIE Mean Width", "mean")]
    ax2.plot(noise_scales, width_mean, color=color, marker="s", linestyle="--", label="Interval Width")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"MAPIE Conformal Adaptation to Label Noise ({dataset_name})", fontsize=12)
    fig.tight_layout()
    ziel_datei = ziel_ordner / f"rq3_mapie_noise_injection_{dataset_name}{title}_{noise_mode_str}.png"
    os.makedirs(os.path.dirname(ziel_datei), exist_ok=True)
    plt.savefig(ziel_datei, dpi=300)
    plt.close()

def inject_noise(y, X=None, noise_scale=1.0, mode="constant", feature_idx=0):
    """Injects synthetic homoscedastic or heteroscedastic Gaussian noise into a target vector.

    Parameters
    ----------
    y : array-like or pandas.Series of shape (n_samples,)
        Target vector to be corrupted with noise.
    X : array-like or pandas.DataFrame of shape (n_samples, n_features), optional
        Feature matrix required when ``mode="growing_feature"`` to determine feature-dependent noise levels.
    noise_scale : float, default=1.0
        Scaling factor multiplied by the standard deviation of `y` to control overall noise magnitude.
    mode : {"constant", "growing_feature", "growing_target"}, default="constant"
        Noise generation scheme:

        - ``"constant"``: Homoscedastic noise with equal variance across all samples.
        - ``"growing_feature"``: Heteroscedastic noise proportional to the scaled values of a selected feature in `X`.
        - ``"growing_target"``: Heteroscedastic noise proportional to the magnitude of target values in `y`.
    feature_idx : int, default=0
        Column index of the feature in `X` used to scale noise when ``mode="growing_feature"``.

    Returns
    -------
    y_noisy : numpy.ndarray of shape (n_samples,)
        Target array with additive Gaussian noise.

    Raises
    ------
    ValueError
        If ``mode="growing_feature"`` and `X` is not provided, or if an unrecognized `mode` string is specified.
    """
    y_std = np.std(y)
    n_samples = len(y)
    
    if mode == "constant":
        sigma = np.full(n_samples, noise_scale * y_std)
        
    elif mode == "growing_feature":
        if X is None:
            raise ValueError("X muss übergeben werden für 'growing_feature' noise.")
        
        # Extrahiere Feature und normalisiere auf [0, 1]
        x_col = X.iloc[:, feature_idx].values if hasattr(X, "iloc") else X[:, feature_idx]
        x_norm = (x_col - x_col.min()) / (x_col.max() - x_col.min() + 1e-8)
        
        # Rauschen wächst linear von 0.1x bis 2.0x der Basis-Standardabweichung
        sigma = noise_scale * y_std * (0.1 + 1.9 * x_norm)
        
    elif mode == "growing_target":
        y_vals = y.values if hasattr(y, "values") else y
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)
        sigma = noise_scale * y_std * (0.1 + 1.9 * y_norm)
        
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")

    noise = np.random.normal(loc=0.0, scale=sigma)
    return y + noise

#run_rq3_experiment(n_sample_sizes=[25],noise_mode="constant")
#run_rq3_experiment(scenario="tabarena")

