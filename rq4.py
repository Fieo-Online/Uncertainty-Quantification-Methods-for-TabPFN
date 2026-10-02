from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from mapie.regression import ConformalizedQuantileRegressor
from tabpfn.constants import ModelVersion
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openml, time, os
import rq1, rq2, rq3
from datetime import datetime
from pathlib import Path
from src import utils, dataloader, uq_Methods
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
os.environ["HF_TOKEN"] = "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"
#TabPFNv2 & v1 maybe, TabICL 
datasets = dataloader.generate_scenario_data("clean", 1000)
skript_ordner = Path(__file__).parent
ziel_ordner = skript_ordner / "results" / "rq4" 

def run_rq4_1_experiment(scenario_name="tabarena", n_samples=1000, cal_ratio=0.3, model="TabPFN_v3"):
    """Runs the Research Question 1 (RQ1) experiment comparing multiple models prediction intervals 
        against MAPIE conformalized prediction intervals across multiple confidence levels.
    
        Parameters
        ----------
        scenario_name : str, default="hetero"
            Name of the data scenario to evaluate (e.g., "hetero", "tabarena", "data_gap").
        n_samples : int, default=400
            Number of synthetic samples generated per dataset (only applicable when scenario_name="hetero").
        cal_ratio : float, default=0.3
            Proportion of the training split reserved as calibration data for MAPIE conformalization.
        model : str, default="TabPFN_v3"
            Initializes chosen model via. "get_model(model)"
        Returns
        -------
        None
            Appends metrics to 'rq4_1results.csv', renders and saves coverage plots to disk, and prints summary tables.
        """
    print(f"==================================================")
    print(f" Running RQ4_1 Evaluation on Scenario: '{scenario_name}'")
    print(f"==================================================\n")

    #Define confidence levels for testing (50%, 80%, 90%, 95%)
    target_confidence_levels = [0.50, 0.80, 0.9]
    alphas = [1.0 - c for c in target_confidence_levels]

    datasets = dataloader.generate_scenario_data(
        scenario_name, n_samples=n_samples
    )
    summary_results = []

    for (X,y,dataset_name) in datasets:

        if len(X) > 100 and X.shape[1] > 15:
            X = X.sample(n=100)
            y = y.loc[X.index]
            print(f"{dataset_name} has to many samples and features so the training size is reduced to 100.")
        elif len(X) > 500:
            X = X.sample(n=500)
            y = y.loc[X.index]
            print(f"{dataset_name} has to many samples so the training size is reduced to 500.")
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
            )
        
        if scenario_name in ["hetero","data_gap"]:
            X_test = X_test.sort_values(by="feature_1")
            y_test = y_test.loc[X_test.index]

        print(
            f"Evaluating: {dataset_name} (Train total: {len(X_train)}, Test size:"
            f" {len(X_test)}, Features: {X_train.shape[1]})"
        )

        y_true_arr = (
            y_test.values.ravel()
            if hasattr(y_test, "values")
            else np.asarray(y_test).ravel()
        )
    
        #Split training data into fit & calibration sets for MAPIE
        X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
            X_train, y_train, test_size=cal_ratio
        )

        #If too big, but leave X_cal and y_cal alone for better MAPPIE results
        if len(X_train_fit) > 200 and X_train_fit.shape[1] > 15:
                    X_train_fit = X_train_fit.sample(n=200)
                    y_train_fit = y_train_fit.loc[X_train_fit.index]
                    print(f"{dataset_name} has to many samples and features so the training size is reduced to 200.")
        elif len(X_train_fit) > 2000:
            X_train_fit = X_train_fit.sample(n=2000)
            y_train_fit = y_train_fit.loc[X_train_fit.index]
            print(f"{dataset_name} has to many samples so the training size is reduced to 2000.")
        
        regressor = get_model(model)
        regressor.fit(X_train_fit, y_train_fit)

        

        #Evaluate coverage at different alphas (Coverage/confidence rates)
        cov_ppd_list, cov_mapie_list = [], []

        # Preparation for plotting results
        X_test_list = []
        X_train_list = []
        y_train_list = []
        y_pred_list = []
        mapie_low_list = []
        mapie_high_list = []
        ppd_low_list = []
        ppd_high_list = []
        conf_level_list = []
    
        for conf_level, alpha in zip(target_confidence_levels, alphas):
            q_low = alpha / 2.0
            q_high = 1.0 - alpha / 2.0
            model_low = utils.QuantileTabPFNAdapter(regressor, alpha=q_low)
            model_high = utils.QuantileTabPFNAdapter(regressor, alpha=q_high)
            model_mid = utils.QuantileTabPFNAdapter(regressor)
            mapie_model = ConformalizedQuantileRegressor(estimator=[model_low, model_high, model_mid], prefit=True, confidence_level=conf_level)
            mapie_model.conformalize(X_cal, y_cal)

            #TabPFN PPD interval calculation with the correct quantiles
            ppd_quantiles = regressor.predict(
                X_test,
                output_type="quantiles",
                quantiles=[q_low, 0.5, q_high],
            )
            ppd_low, ppd_high = ppd_quantiles[0][:], ppd_quantiles[2][:]

            metrics_ppd = evaluate_interval_quality(
                y_true_arr, ppd_low, ppd_high, alpha=alpha
            )

            #MAPIE Conformal interval calculation
            y_pred, y_pis = mapie_model.predict_interval(X_test)
            mapie_low = y_pis[:, 0, 0]
            mapie_high = y_pis[:, 1, 0]

            metrics_mapie = evaluate_interval_quality(
                y_true_arr, mapie_low, mapie_high, alpha=alpha
            )

            cov_ppd_list.append(metrics_ppd["coverage_rate"])
            cov_mapie_list.append(metrics_mapie["coverage_rate"])

            summary_results.append({
                "Dataset": dataset_name,
                "Nominal Confidence": f"{int(conf_level * 100)}%",
                "PPD Coverage": f"{metrics_ppd['coverage_rate'] * 100:.1f}%",
                "PPD Width": f"{metrics_ppd['mean_width']:.3f}",
                "PPD Winkler": f"{metrics_ppd['winkler_score']:.3f}",
                "MAPIE Coverage": (
                    f"{metrics_mapie['coverage_rate'] * 100:.1f}%"
                ),
                "MAPIE Width": f"{metrics_mapie['mean_width']:.3f}",
                "MAPIE Winkler": f"{metrics_mapie['winkler_score']:.3f}",
            })

            X_test_list.append(X_test)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            y_pred_list.append(y_pred)
            mapie_low_list.append(mapie_low)
            mapie_high_list.append(mapie_high)
            ppd_low_list.append(ppd_low)
            ppd_high_list.append(ppd_high)
            conf_level_list.append(conf_level)    

        #Visualizing the Plots and the interval coverage for synthetic datasets
        packs = [X_test_list, X_train_list, y_train_list, y_pred_list, mapie_low_list, mapie_high_list, ppd_low_list, ppd_high_list, conf_level_list]
        if scenario_name in ["hetero","data_gap"]:
            utils.coverage_plot(len(target_confidence_levels), packs,title=f"{conf_level*100}% Coverage MAPPIE VS. PPD- {dataset_name}{model}")

        #Visualization for the 95% confidence interval MAPPIE
        y_pred_90, y_pis_90 = mapie_model.predict_interval(X_test)
        mapie_low_90 = y_pis_90[:, 0, 0]
        mapie_high_90 = y_pis_90[:, 1, 0]
        intervals_90 = np.column_stack([mapie_low_90, mapie_high_90])

        #Visualization for the 95% confidence interval TabPFNA
        ppd_90 = regressor.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        ppd_low_90, ppd_med_90, ppd_high_90 = ppd_quantiles[0][:], ppd_quantiles[1][:], ppd_quantiles[2][:]
        intervals_90 = np.column_stack([ppd_low_90, ppd_high_90])

        

    # Output summary table
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv('rq4_1results.csv', index=False,mode='a')
    print("\n================ FINAL RQ1 SUMMARY TABLE ================")
    print(df_summary.to_string(index=False))
    print("=========================================================\n")

def get_model(model_name: str):
    """Instantiates and returns a tabular regression model instance based on the specified model identifier.
        Models have to saved in the tabpfn folder to be used.
    Parameters
    ----------
    model_name : {"TabPFN_v2", "TabPFN_v2.5", "TabPFN_v3", "TabICL"}
        Name of the requested tabular regression model variant:

        - ``"TabPFN_v2"``: TabPFN regressor initialized with the v2 checkpoint.
        - ``"TabPFN_v2.5"``: TabPFN regressor initialized with the v2.5 default checkpoint.
        - ``"TabPFN_v3"``: TabPFN regressor initialized with the default v3 settings.
        - ``"TabICL"``: TabICL regressor instance.

    Returns
    -------
    TabPFNRegressor or TabICLRegressor
        An initialized regressor model instance ready for training and prediction.

    Raises
    ------
    ValueError
        If `model_name` does not match any of the supported model identifiers.
    """
    if model_name == "TabPFN_v2":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", ignore_pretraining_limits=True)
        
    elif model_name == "TabPFN_v2.5":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(model_path="tabpfn-v2.5-regressor-v2.5_default.ckpt",ignore_pretraining_limits=True)

    elif model_name == "TabPFN_v3":
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor(ignore_pretraining_limits=True)
         
    
    elif model_name == "TabICL":

        return TabICLRegressor()
        
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

def evaluate_interval_quality(y_true, lower_bounds, upper_bounds, alpha=0.10):
    """Calculates key uncertainty quantification (UQ) metrics for prediction intervals, including coverage rate, mean width, and Winkler Score.
    
        Parameters
        ----------
        y_true : array-like
            Ground-truth target values.
        lower_bounds : array-like
            Lower bounds of the prediction intervals.
        upper_bounds : array-like
            Upper bounds of the prediction intervals.
        alpha : float, default=0.10
            Significance level corresponding to the nominal coverage rate of 1 - alpha 
            (e.g., alpha=0.10 for a 90% prediction interval).
    
        Returns
        -------
        metrics : dict
            Dictionary containing calculated interval evaluation metrics:
    
            - ``"coverage_rate"``: float
                Proportion of ground-truth values falling within the prediction intervals.
            - ``"mean_width"``: float
                Average width of the prediction intervals across all samples.
            - ``"winkler_score"``: float
                Mean Winkler Score, penalizing both interval width and coverage breaches.
            - ``"covered_mask"``: numpy.ndarray
                Boolean array indicating whether each sample was successfully covered by its interval.
        """
    y_true = np.asarray(y_true).ravel()
    lower_bounds = np.asarray(lower_bounds).ravel()
    upper_bounds = np.asarray(upper_bounds).ravel()

    #Coverage Rate
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage_rate = np.mean(covered)

    #Mean Interval Width
    widths = upper_bounds - lower_bounds
    mean_width = np.mean(widths)

    #Winkler Score (Scoring rule for intervals)
    #Penalizes both wide intervals and target values falling outside the interval
    penalty_below = (2 / alpha) * (lower_bounds - y_true) * (y_true < lower_bounds)
    penalty_above = (2 / alpha) * (y_true - upper_bounds) * (y_true > upper_bounds)
    winkler_scores = widths + penalty_below + penalty_above
    mean_winkler = np.mean(winkler_scores)

    return {
        "coverage_rate": coverage_rate,
        "mean_width": mean_width,
        "winkler_score": mean_winkler,
        "covered_mask": covered
    }


def rq4_2_experiment(model="TabPFN_v3"):
    """Runs the Research Question 2 (RQ2) experiment evaluating Martingale Posterior uncertainty quantification across datasets.
    
        This function loads datasets for a specified experimental scenario, applies subsampling constraints for computational efficiency,
        evaluates performance using 5-fold cross-validation, and generates fold-level MSE vs. uncertainty and sparsification plots.
        Additionally, it executes a single train-test split run per dataset to produce multi-dataset comparative plots for the MSE vs. uncertainty plots.
    
        Parameters
        ----------
        model : str, default="TabPFN_v3"
            Name of the model that has to be evaluated.
    
        Returns
        -------
        None
            Renders and saves evaluation plots (MSE vs. UQ and sparsification curves) directly to disk.
        """
    datasets = dataloader.generate_scenario_data("tabarena", 1000)

    for X, y, name in datasets:
        print(f"For {name}:")

        if len(X) > 100 and X.shape[1] > 15:
            X = X.sample(n=100)
            y = y.loc[X.index]
            print(f"{name} has to many samples and features so the training size is reduced to 100.")
        elif len(X) > 500:
            X = X.sample(n=500)
            y = y.loc[X.index]
            print(f"{name} has to many samples so the training size is reduced to 500.")

        
        time1 = time.time()
        folds = run_5fold_cv_for_dataset(X, y, B=100, T_fwd=50, K=99,model=model) #[y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid]
        print(f"Time for 5Fold Run: {time.time()-time1}")
        #Get clean lists of the different inputs
        y_trues = [fold[0] for fold in folds]
        y_preds = [fold[1] for fold in folds]
        uncs    = [fold[2] for fold in folds]
        pos     = [fold[3] for fold in folds]
        y_grids = [fold[4] for fold in folds]
        
        
        utils.mse_plot_multi_folds(y_true_list=y_trues, y_pred_list=y_preds, uncertainties_list=uncs,dataset_names=[f"Fold {i}" for i in range(len(folds))],title=f"{name}{model}")
        utils.plot_sparsification_multi_sets(y_trues,y_preds,uncs,dataset_names=[f"Fold {i}" for i in range(len(folds))], sigma= 6, title=f"{name}{model}")

def run_5fold_cv_for_dataset(X, y, B=50, T_fwd=50, K=99, device="cpu", model="TabPFN_v3"):
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
            X_train, y_train, X_test, y_test,model=model, B=B, T_fwd=T_fwd, K=K, device=device,
        )

        folds.append([y_test_fold, y_pred_median_fold, uq_fold, posterior, y_grid])

    return folds

def evaluate_single_run(X_train, y_train, X_test, y_test, B=50, T_fwd=50, K=99, device="cpu", model="TabPFN_v3"):
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
    model = get_model(model)
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

def rq4_inference():
    """Measures and benchmarks inference latency across tabular regression models.

    Evaluates various model architectures (TabPFN variants and TabICL) on benchmark
    datasets from the ``"tabarena"`` scenario by fitting once, performing a warm-up prediction,
    and timing repeated inference calls using high-resolution performance counters.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        DataFrame containing inference benchmarking results with the following columns:

        - **"Dataset"** (*str*): Name of the dataset evaluated.
        - **"Model"** (*str*): Model identifier (e.g., ``"TabPFN_v2"``, ``"TabICL"``).
        - **"N_train"** (*int*): Fixed number of training samples used during fitting.
        - **"N_test"** (*int*): Number of test samples evaluated during inference.
        - **"Mean_Inference_Time"** (*float*): Mean execution time per inference call in seconds.
        - **"Median_Inference_Time"** (*float*): Median execution time per inference call in seconds.
        - **"Std_Inference_Time"** (*float*): Standard deviation of execution times across repetitions.
    """

    regressor_v2 = get_model("TabPFN_v2")
    regressor_v2_5 = get_model("TabPFN_v2.5")
    regressor_v3 = get_model("TabPFN_v3")
    regressor_icl = get_model("TabICL")

    regressors = {
        "TabPFN_v2": regressor_v2,
        "TabPFN_v2.5": regressor_v2_5,
        "TabPFN_v3": regressor_v3,
        "TabICL": regressor_icl,
    }

    datasets = dataloader.generate_scenario_data("tabarena")
    results = []

    n_train = 100
    n_repetitions = 5

    for X, y, name in datasets[5:]:

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_test = X[n_train:]
        print(f"Starting {name}")
        for model_name, regressor in regressors.items():

            # Fit model once; fitting time is not part of inference time
            print(f"Fitting {model_name}")
            regressor.fit(X_train, y_train)

            # Warm-up prediction
            print("Warmup")
            _ = regressor.predict(X_test)

            times = []

            for _ in range(n_repetitions):
                print(f"Run {n_repetitions}")
                start = time.perf_counter()

                __ = regressor.predict(X_test)

                end = time.perf_counter()

                times.append(end - start)
                print(end-start)

            results.append({
                "Dataset": name,
                "Model": model_name,
                "N_train": n_train,
                "N_test": len(X_test),
                "Mean_Inference_Time": np.mean(times),
                "Median_Inference_Time": np.median(times),
                "Std_Inference_Time": np.std(times),
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "rq4_inference_results.csv",
        index=False,
        mode = 'a'
    )

    return results_df


    