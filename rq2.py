from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import openml, time, os
from datetime import datetime
from src import utils, dataloader, uq_Methods
from scipy.stats import spearmanr

os.environ["TABPFN_TOKEN"] = "Your PriorLabs TabPFN Token"

def get_cdf(y_grid, preds, quantiles):
    """Interpolates predicted quantile values onto a target grid to construct cumulative distribution function (CDF) curves.
    Used for the martingale posterior function "get_posterior".
    Parameters
    ----------
    y_grid : array-like of shape (K,)
        1D grid of evaluation values along the target response variable space.
    preds : array-like of shape (n_quantiles, n_eval)
        Predicted response values corresponding to each quantile for every evaluation sample.
    quantiles : array-like of shape (n_quantiles,)
        Cumulative probability thresholds (e.g., values from 0.01 to 0.99) associated with each row in `preds`.

    Returns
    -------
    cdf_conditionals : numpy.ndarray of shape (K, n_eval)
        Matrix of evaluated cumulative probability values bounded between 0.0 and 1.0, 
        where each column represents the estimated conditional CDF for a sample over `y_grid`.
    """
    #print("Transforming CDF...")
    preds = np.array(preds) 
    K = len(y_grid)
    n_eval = preds.shape[1]
    time1 = time.time()
    cdf_conditionals = np.zeros((K, n_eval))
    for i in range(n_eval):
        cdf_conditionals[:, i] = np.interp(
            y_grid, 
            xp=preds[:, i],    # Die von TabPFN vorhergesagten Y-Werte
            fp=quantiles,      # Die dazugehörigen Wahrscheinlichkeiten (0.01 bis 0.99)
            left=0.0, right=1.0       # Standardwerte außerhalb des Bereichs
        )
    #print(f"Transformation finished in: {time1-time.time()}")
    return cdf_conditionals

def mp_runs_conv(y_grid, cdf_conditionals, y_train, X_test,  B=[10,25,50,100], K=[50,100,200,500]):
    """Running the different setting for get_posterior to see how the MP developes with growing B's and K's.
        runs: [ppd, (b,k)]
    """
    print("Getting the MP's for different B's and K's...")

    time1 = time.time()
    B_runs = []
    #Getting MP for every B-Value in the B-list
    for b in B:
        time2 = time.time()
        ppd = uq_Methods.get_posterior(y_grid,len(y_train),b,50,cdf_conditionals,alpha_dimension=X_test.shape[1])
        B_runs.append((ppd,(b,50)))
        print(f"MP with B = {b} and K = {50} finished in {time2-time.time()}")
    print(f"MP's for B were finished in {time1 - time.time()}") 

    time1 = time.time()
    K_runs = []
    #Getting MP for every T_fwd-Value in the K-list
    for k in K:
        time2 = time.time()
        ppd = uq_Methods.get_posterior(y_grid,len(y_train),50,k,cdf_conditionals,alpha_dimension=X_test.shape[1])
        K_runs.append((ppd,(50,k)))
        print(f"MP with B = {b} and K = {50} finished in {time2-time.time()}")
    print(f"MP's for K were finished in {time1 - time.time()}") 

    runs_uncertainties = []
    labels = []
    
    for ppd, (b, k) in B_runs:
        # Calculate uncertainty for the B Runs
        uq_dic = uq_Methods.total_martingale_uq(ppd, len(y_train), y_grid)
        uq = uq_dic["total_uncertainty_var"]
        runs_uncertainties.append(uq)
        labels.append(f"B = {b}, K = {k}")

    for ppd, (b, k) in K_runs:
            # Calculate uncertainty for the K Runs
            uq_dic = uq_Methods.total_martingale_uq(ppd, len(y_train), y_grid)
            uq = uq_dic["total_uncertainty_var"]
            runs_uncertainties.append(uq)
            labels.append(f"B = {b}, K = {k}")    

    #Plot the results
    utils.plot_mp_convergence(
        B_runs=B_runs,
        K_runs=K_runs
    )
    
def run_5fold_cv_for_dataset(X, y, B=50, T_fwd=50, K=99, device="cpu"):
    """Performs 5-fold cross-validation on a dataset to evaluate predictions and uncertainty quantification metrics.

    Parameters
    ----------
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix for the dataset.
    y : array-like or pandas.Series of shape (n_samples,)
        Target vector corresponding to features in `X`.
    B : int, default=50
        Number of bootstrap samples or ensemble repetitions evaluated per fold.
    T_fwd : int, default=50
        Number of forward passes or stochastic evaluations per test sample.
    K : int, default=99
        Number of grid points constructed along the evaluation target space.
    device : str, default="cpu"
        Compute device allocated for inference processing (e.g., ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    folds : list of list
        Outer list containing 5 inner lists (one per cross-validation fold). 
        Each inner list contains:

        - **y_test_fold** (*array-like*): Ground-truth target values for the test fold.
        - **y_pred_median_fold** (*array-like*): Median point predictions for the test fold.
        - **uq_fold** (*array-like*): Uncertainty estimates (e.g., variances or interval widths) for the test fold.
        - **posterior** (*array-like*): Estimated posterior probability densities across `y_grid`.
        - **y_grid** (*array-like*): 1D array of target grid evaluation points used for posterior estimation.
    """
    folds = []
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    #5 Fold cross validator, gives indices for splits
    kf = KFold(n_splits=5, shuffle=True)

    #Takes the 5 different train-test splits and performs the runs (aka. predication and UQ-Method)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid, _= evaluate_single_run(
            X_train, y_train, X_test, y_test, B=B, T_fwd=T_fwd, K=K, device=device
        )

        folds.append([y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid])

    return folds

def evaluate_single_run(X_train, y_train, X_test, y_test, B=50, T_fwd=50, K=99, device="cpu"):
    """Fits TabPFN on training data, estimates conditional CDFs, and computes Martingale Posterior total uncertainty.

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
    B : int, default=50
        Number of posterior bootstrap samples used in the Martingale Posterior calculation.
    T_fwd : int, default=50
        Number of forward predictive samples generated per test instance.
    K : int, default=99
        Number of evaluation points along the response grid and quantile spectrum.
    device : str, default="cpu"
        Computation device allocated for model inference (e.g., ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    y_test : numpy.ndarray of shape (n_test_samples,)
        Array of ground-truth test target values.
    preds_median : numpy.ndarray of shape (n_test_samples,)
        Array of median predictions (50th percentile) generated by TabPFN.
    uncertainties : numpy.ndarray of shape (n_test_samples,)
        Total uncertainty estimates derived from the Martingale Posterior variances.
    post : numpy.ndarray
        Computed Martingale Posterior distribution densities evaluated across `y_grid`.
    y_grid : numpy.ndarray of shape (K,)
        1D array of uniformly spaced target values spanning the range ``[y_train.min(), y_train.max()]``.
    cdf_conditionals : numpy.ndarray of shape (K, n_test_samples)
        Interpolated conditional cumulative distribution function values for test instances over `y_grid`.
    """
    # TabPFN training
    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
    model.fit(X_train, y_train)
    
    #Only median prediction
    preds_median = model.predict(X_test, output_type="quantiles", quantiles=[0.5])
    
    y_min, y_max = y_train.min(), y_train.max()
    y_grid = np.linspace(y_min, y_max, K)
    quantiles_grid = np.linspace(0.01, 0.99, K)

    preds = model.predict(X_test, output_type="quantiles", quantiles=quantiles_grid)

    #Getting the right inputs to calculate MP
    cdf_conditionals = get_cdf(y_grid,preds,quantiles_grid)
    n_train = len(X_train)
    d = X_train.shape[1]
    
    post = uq_Methods.get_posterior(
        y_grid=y_grid,
        n_train=n_train,
        B_postsamples=B,
        T_fwdsamples=T_fwd,
        cdf_conditionals=cdf_conditionals,
        alpha_dimension=d,
        use_blowup=True
    )

    uq_dic = uq_Methods.total_martingale_uq(post, len(X_train), y_grid)
    uncertainties = uq_dic["total_uncertainty_var"]
    
    return np.array(y_test), np.array(preds_median), np.array(uncertainties), post, y_grid, cdf_conditionals

def rq2_experiment(scenario="hetero"):
    """Runs the Research Question 2 (RQ2) experiment evaluating Martingale Posterior uncertainty quantification across datasets.

    This function loads datasets for a specified experimental scenario, applies subsampling constraints for computational efficiency,
    evaluates performance using 5-fold cross-validation, and generates fold-level MSE vs. uncertainty and sparsification plots.
    Additionally, it executes a single train-test split run per dataset to produce multi-dataset comparative plots for the MSE vs. uncertainty plots.

    Parameters
    ----------
    scenario : str, default="hetero"
        Name of the data scenario to evaluate (e.g., ``"hetero"``, ``"tabarena"``, or ``"data_gap"``).

    Returns
    -------
    None
        Renders and saves evaluation plots (MSE vs. UQ and sparsification curves) directly to disk.
    """
    datasets = dataloader.generate_scenario_data(scenario_name=scenario)
    runs = []
    dataset_names = [name for _, _, name in datasets]
    for X, y, name in datasets:
        print(f"For {name}:")
        
        if len(X) > 500 and X.shape[1] > 15:
            X = X.sample(n=500)
            y = y.loc[X.index]
            print(f"{name} has to many samples and features so the training size is reduced to 500.")
        elif len(X) > 2000:
            X = X.sample(n=2000)
            y = y.loc[X.index]
            print(f"{name} has to many samples so the training size is reduced to 2000.")
                
        time1 = time.time()
        folds = run_5fold_cv_for_dataset(X, y, B=100, T_fwd=50, K=99) #[y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid]
        print(f"Time for 5Fold Run: {time.time()-time1}")
        #Get clean lists of the different inputs
        y_trues = [fold[0] for fold in folds]
        y_preds = [fold[1] for fold in folds]
        uncs    = [fold[2] for fold in folds]
        pos     = [fold[3] for fold in folds]
        y_grids = [fold[4] for fold in folds]
        
        
        utils.mse_plot_multi_folds(y_true_list=y_trues, y_pred_list=y_preds, uncertainties_list=uncs,dataset_names=[f"Fold {i}" for i in range(len(folds))], title=f"MSE vs. UQ {name}")
        utils.plot_sparsification_multi_sets(y_trues,y_preds,uncs,dataset_names=[f"Fold {i}" for i in range(len(folds))], sigma= 2,title=f"MSE Sparsification Plot {name}")
        
        #Preparation for MSE-plot of all datasets
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
        )
        #np.array(y_test), np.array(preds_median), np.array(uncertainties), post, y_grid, cdf_conditionals
        y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid, _= evaluate_single_run(
                    X_train, y_train, X_test, y_test, B=50, T_fwd=50
                )
        runs.append([y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid]) #y_test, preds_median, uncertainties, post, y_grid

    #Get clean lists of the different inputs
    y_trues = [fold[0] for fold in runs]
    y_preds = [fold[1] for fold in runs] 
    uncs    = [fold[2] for fold in runs]
    pos     = [fold[3] for fold in runs]
    y_grids = [fold[4] for fold in runs]
    #MSE and Spars plot of every dataset (MSE plot is shown as a grid, because of different MSE)
    utils.mse_plot_multi_sets(y_trues,y_preds,uncs,dataset_names, title=scenario)
    utils.plot_sparsification_multi_sets(y_trues,y_preds,uncs,dataset_names,False,sigma=2,title=f"MSE Sparsification Plot {scenario}")

    """
    #MP convergence
    for dataset in datasets:
        X_test, y_test, preds, y_train = training.fit_and_predict(datasets[dataset][0],datasets[dataset][1], quantiles=np.linspace(0.01,0.99,99),output_type="quantiles", test_size1=0.2)

        y_min = float(np.min(y_train))
        y_max = float(np.max(y_train))
        y_grid = np.linspace(y_min,y_max,99)   

        quantiles = np.linspace(0.01, 0.99, 99) 
        cdf_conditionals = get_cdf(y_grid, preds,quantiles)
        mp_runs_conv(y_grid,cdf_conditionals, y_train, X_test,B= np.linspace(1,100,10,dtype=int), K=np.linspace(1,500,20,dtype=int))
    """
    
def generate_scenario_data(mode="heteroscedastic_only", n_samples=250):
    """
    Generates synthetic data for controlled UQ scenarios.
    
    Scenario A ('heteroscedastic_only'): Strong noise variation, NO gap.
    Scenario B ('gap_only'): Constant noise, with Data Gap.
    """
    gap_bounds = (-1, 1)
    x_test = np.linspace(-3.2, 3.2, 250)
    y_true_test = x_test**2 - 1.0

    if mode == "heteroscedastic_only":
        # Scenario A: Dense data, increasing noise
        x_train = np.random.uniform(-2.5, 2.5, size=n_samples)
        noise_std_train = 0.05 + 0.25 * (x_train + 2.5)
        y_train = (x_train**2 - 1.0) + np.random.normal(0, noise_std_train)
        
        true_noise_std = 0.05 + 0.25 * (x_test + 2.5)
        gap_bounds = None  # No gap present

    elif mode == "gap_only":
        # Scenario B: Data Gap, constant noise
        x_raw = np.random.uniform(-2.5, 2.5, size=n_samples)
        mask = (x_raw < gap_bounds[0]) | (x_raw > gap_bounds[1])
        x_train = x_raw[mask]
        
        noise_std_train = 0.25  # Constant noise
        y_train = (x_train**2 - 1.0) + np.random.normal(0, noise_std_train, size=len(x_train))
        
        true_noise_std = np.full_like(x_test, 0.25)

    return (
        x_train.reshape(-1, 1),
        y_train,
        x_test.reshape(-1, 1),
        y_true_test,
        true_noise_std,
        gap_bounds,
    )

def compute_uq_for_scenario(X_train, y_train, X_test, B=50, T_fwd=50, K=99):
    """Fits TabPFN on training data and performs epistemic/aleatoric uncertainty decomposition via the Law of Total Variance 
    using the function "total_martingale_uq".
    Used for controlled uncertainty disentanglement comparisson in "run_2x2_comparison_plot"

    Parameters
    ----------
    X_train : pandas.DataFrame or array-like of shape (n_train_samples, n_features)
        Feature matrix for training data.
    y_train : pandas.Series or array-like of shape (n_train_samples,)
        Target values for training data.
    X_test : pandas.DataFrame or array-like of shape (n_test_samples, n_features)
        Feature matrix for test evaluation.
    B : int, default=50
        Number of posterior bootstrap samples drawn for Martingale Posterior simulation.
    T_fwd : int, default=50
        Number of forward predictive samples generated per test instance.
    K : int, default=99
        Number of grid points constructed along the evaluation target space and quantile spectrum.

    Returns
    -------
    dict
        Dictionary containing predicted quantiles and uncertainty estimates with keys:

        - **"preds_quantiles"** (*numpy.ndarray* of shape `(K, n_test_samples)`): Predicted quantile matrix from TabPFN.
        - **"aleatoric"** (*numpy.ndarray* of shape `(n_test_samples,)`): Aleatoric uncertainty variance per test sample.
        - **"epistemic"** (*numpy.ndarray* of shape `(n_test_samples,)`): Epistemic uncertainty variance per test sample.
        - **"total"** (*numpy.ndarray* of shape `(n_test_samples,)`): Total uncertainty variance (sum of aleatoric and epistemic).
        - **"var_ppd"** (*numpy.ndarray* of shape `(n_test_samples,)`): Variance directly calculated from the TabPFN posterior predictive distribution.
    """
    n_train = len(X_train)
    n_test = len(X_test)

    # 1. TabPFN Modeling
    model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    model.fit(X_train, y_train)

    quantiles_grid = np.linspace(0.01, 0.99, K)
    y_min, y_max = y_train.min() - 2.0, y_train.max() + 2.0
    y_grid = np.linspace(y_min, y_max, K)

    preds_quantiles = model.predict(X_test, output_type="quantiles", quantiles=quantiles_grid)

    cdf_conditionals = get_cdf(y_grid, preds_quantiles, quantiles_grid)

    # 2. Martingale Posterior
    post = uq_Methods.get_posterior(
        y_grid=y_grid,
        n_train=n_train,
        B_postsamples=B,
        T_fwdsamples=T_fwd,
        cdf_conditionals=cdf_conditionals,
        alpha_dimension=1,
        use_blowup=True,
    )

    # 3. Variance calculations
    var_ppd = np.zeros(n_test)

    for j in range(n_test):
        # Base TabPFN PPD variance
        cdf_ppd = cdf_conditionals[:, j]
        pdf_ppd = np.diff(cdf_ppd, prepend=0)
        pdf_ppd /= np.maximum(np.sum(pdf_ppd), 1e-8)
        mean_ppd = np.sum(pdf_ppd * y_grid)
        var_ppd[j] = np.sum(pdf_ppd * (y_grid - mean_ppd) ** 2)

        # Martingale Posterior path variances
        uq = uq_Methods.total_martingale_uq(post,n_train, y_grid)

    return {
        "preds_quantiles": preds_quantiles,
        "aleatoric": uq["aleatoric_uncertainty_var"],
        "epistemic": uq["epistemic_uncertainty_var"],
        "total": uq["total_uncertainty_var"],
        "var_ppd": var_ppd,
    }

def run_2x2_comparison_plot():
    """Creates a 2x2 comparisson plot for two Scenarios. Used to watch the uncertainty disentanglement."""
    print("Starting 2x2 comparison experiment (Scenario A vs. Scenario B)...")

    # --- Compute data & UQ ---
    # Scenario A
    X_tr_A, y_tr_A, X_te, y_true_te, noise_std_A, gap_A = generate_scenario_data("heteroscedastic_only")
    res_A = compute_uq_for_scenario(X_tr_A, y_tr_A, X_te)
    summary = {
        "scenario": "heteroscedastic_only",
        "all_aleatoric": res_A["aleatoric"],
        "all_epiostemic": res_A["epistemic"],
        "mean_aleatoric": float(np.mean(res_A["aleatoric"])),
        "max_aleatoric": float(np.max(res_A["aleatoric"])),
        "mean_epistemic": float(np.mean(res_A["epistemic"])),
        "max_epistemic": float(np.max(res_A["epistemic"])),
        "mean_total": float(np.mean(res_A["total"])),
        "aleatoric increase":float(np.max(res_A["aleatoric"]))/float(np.min(res_A["aleatoric"])),
        "epistemic_increase": float(np.max(res_A["epistemic"]))/float(np.min(res_A["epistemic"])),
        "epistemic_ratio_mean": float(np.mean(res_A["epistemic"] / (res_A["total"] + 1e-8))),
        "aleatoric_ratio_mean": float(np.mean(res_A["aleatoric"] / (res_A["total"] + 1e-8)))
    }
    print(summary)
    # Scenario B
    X_tr_B, y_tr_B, _, _, noise_std_B, gap_B = generate_scenario_data("gap_only")
    res_B = compute_uq_for_scenario(X_tr_B, y_tr_B, X_te)
    summary = {
            "scenario": "gap_only",
            "all_aleatoric": res_B["aleatoric"],
            "all_epiostemic": res_B["epistemic"],
            "mean_aleatoric": float(np.mean(res_B["aleatoric"])),
            "max_aleatoric": float(np.max(res_B["aleatoric"])),
            "mean_epistemic": float(np.mean(res_B["epistemic"])),
            "max_epistemic": float(np.max(res_B["epistemic"])),
            "mean_total": float(np.mean(res_B["total"])),
            "aleatoric increase":float(np.max(res_B["aleatoric"]))/float(np.min(res_B["aleatoric"])),
            "epistemic_increase": float(np.max(res_B["epistemic"]))/float(np.min(res_B["epistemic"])),
            "epistemic_ratio_mean": float(np.mean(res_B["epistemic"] / (res_B["total"] + 1e-8))),
            "aleatoric_ratio_mean": float(np.mean(res_B["aleatoric"] / (res_B["total"] + 1e-8)))
        }
    print(summary)

    x_test_flat = X_te.ravel()

    # --- Plotting Grid (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    # =========================================================================
    # COLUMN 1: SCENARIO A (Isolated Aleatoric Uncertainty)
    # =========================================================================
    # Panel (0, 0): Data Fit
    axes[0, 0].scatter(X_tr_A, y_tr_A, color="black", alpha=0.5, s=15, label="Training data")
    axes[0, 0].plot(x_test_flat, y_true_te, "g--", lw=2, label="True f(x)")
    axes[0, 0].plot(x_test_flat, res_A["preds_quantiles"][50], "b-", lw=2, label="TabPFN Median")
    axes[0, 0].fill_between(x_test_flat, res_A["preds_quantiles"][5], res_A["preds_quantiles"][95], color="blue", alpha=0.15, label="TabPFN Base PPD (90% Interval)")
    axes[0, 0].axvspan(-3.2, -2.5, color="gray", alpha=0.1, label="OOD region")
    axes[0, 0].axvspan(2.5, 3.2, color="gray", alpha=0.1)
    axes[0, 0].set_ylabel("Target y")
    axes[0, 0].set_title("Scenario A: Heteroscedastic Noise (Dense Data)")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel (1, 0): Uncertainty Decomposition
    axes[1, 0].plot(x_test_flat, res_A["aleatoric"], color="orange", lw=2.5, label="Aleatoric $\mathbb{E}_B[\mathrm{Var}(Y|B)]$")
    axes[1, 0].plot(x_test_flat, res_A["epistemic"], color="purple", lw=2.5, label="Epistemic $\mathrm{Var}_B(\mathbb{E}[Y|B])$")
    axes[1, 0].plot(x_test_flat, res_A["total"], color="darkred", linestyle="-.", lw=2, label="Martingale Total")
    axes[1, 0].plot(x_test_flat, res_A["var_ppd"], color="blue", linestyle=":", lw=2, label="TabPFN Base PPD Variance")
    axes[1, 0].plot(x_test_flat, noise_std_A**2, "g--", alpha=0.7, label="True noise variance $\sigma^2(x)$")
    axes[1, 0].axvspan(-3.2, -2.5, color="gray", alpha=0.1)
    axes[1, 0].axvspan(2.5, 3.2, color="gray", alpha=0.1)
    axes[1, 0].set_xlabel("Feature x")
    axes[1, 0].set_ylabel("Variance / Uncertainty")
    axes[1, 0].set_title("Decomposition Scenario A: Aleatoric dominated")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)

    # =========================================================================
    # COLUMN 2: SCENARIO B (Isolated Epistemic Uncertainty)
    # =========================================================================
    # Panel (0, 1): Data Fit
    axes[0, 1].scatter(X_tr_B, y_tr_B, color="black", alpha=0.5, s=15, label="Training data")
    axes[0, 1].plot(x_test_flat, y_true_te, "g--", lw=2, label="True f(x)")
    axes[0, 1].plot(x_test_flat, res_B["preds_quantiles"][50], "b-", lw=2, label="TabPFN Median")
    axes[0, 1].fill_between(x_test_flat, res_B["preds_quantiles"][5], res_B["preds_quantiles"][95], color="blue", alpha=0.15, label="TabPFN Base PPD (90% Interval)")
    axes[0, 1].axvspan(gap_B[0], gap_B[1], color="red", alpha=0.15, label="Data Gap")
    axes[0, 1].axvspan(-3.2, -2.5, color="gray", alpha=0.1, label="OOD region")
    axes[0, 1].axvspan(2.5, 3.2, color="gray", alpha=0.1)
    axes[0, 1].set_title("Scenario B: Data Gap (Homoscedastic Noise)")
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel (1, 1): Uncertainty Decomposition
    axes[1, 1].plot(x_test_flat, res_B["aleatoric"], color="orange", lw=2.5, label="Aleatoric $\mathbb{E}_B[\mathrm{Var}(Y|B)]$")
    axes[1, 1].plot(x_test_flat, res_B["epistemic"], color="purple", lw=2.5, label="Epistemic $\mathrm{Var}_B(\mathbb{E}[Y|B])$")
    axes[1, 1].plot(x_test_flat, res_B["total"], color="darkred", linestyle="-.", lw=2, label="Martingale Total")
    axes[1, 1].plot(x_test_flat, res_B["var_ppd"], color="blue", linestyle=":", lw=2, label="TabPFN Base PPD Variance")
    axes[1, 1].plot(x_test_flat, noise_std_B**2, "g--", alpha=0.7, label="True noise variance $\sigma^2$")
    axes[1, 1].axvspan(gap_B[0], gap_B[1], color="red", alpha=0.15)
    axes[1, 1].axvspan(-3.2, -2.5, color="gray", alpha=0.1)
    axes[1, 1].axvspan(2.5, 3.2, color="gray", alpha=0.1)
    axes[1, 1].set_xlabel("Feature x")
    axes[1, 1].set_title("Decomposition Scenario B: Epistemic dominated in Gap/OOD")
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rq2_1_scenarios_comparison_2x2.png", dpi=300)
    plt.show()
    print("Done! Graphic saved under 'rq2_1_scenarios_comparison_2x2.png'.")

#RQ2.1
#run_2x2_comparison_plot()
#RQ2.2
#rq2_experiment()
#rq2_experiment("tabarena")