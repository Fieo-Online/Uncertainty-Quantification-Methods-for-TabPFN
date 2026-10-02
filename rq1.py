from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime
from src import utils, dataloader
from mapie.regression import ConformalizedQuantileRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
   
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

def run_rq1_experiment(scenario_name="hetero", n_samples=400, cal_ratio=0.3):
    """Runs the Research Question 1 (RQ1) experiment comparing TabPFN Posterior Predictive Distribution (PPD) intervals 
    against MAPIE conformalized prediction intervals across multiple confidence levels.

    Parameters
    ----------
    scenario_name : str, default="hetero"
        Name of the data scenario to evaluate (e.g., "hetero", "tabarena", "data_gap").
    n_samples : int, default=400
        Number of synthetic samples generated per dataset (only applicable when scenario_name="hetero").
    cal_ratio : float, default=0.3
        Proportion of the training split reserved as calibration data for MAPIE conformalization.

    Returns
    -------
    None
        Appends metrics to 'rq1results.csv', renders and saves coverage plots to disk, and prints summary tables.
    """
    print(f"==================================================")
    print(f" Running RQ1 Evaluation on Scenario: '{scenario_name}'")
    print(f"==================================================\n")

    #Define confidence levels for testing (50%, 80%, 85%, 90%)
    target_confidence_levels = [0.50, 0.80, 0.85, 0.9]
    alphas = [1.0 - c for c in target_confidence_levels]

    datasets = dataloader.generate_scenario_data(
        scenario_name, n_samples=n_samples
    )
    summary_results = []

    for (X,y,dataset_name) in datasets:
        
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
        
        regressor = TabPFNRegressor(ignore_pretraining_limits=True,)
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

            #CQR with 3 models for each needed Quantile
            mapie_model = ConformalizedQuantileRegressor(estimator=[model_low, model_high, model_mid], prefit=True, confidence_level=conf_level)
            mapie_model.conformalize(X_cal, y_cal)

            #TabPFN PPD interval calculation with the correct quantiles
            ppd_quantiles = regressor.predict(
                X_test,
                output_type="quantiles",
                quantiles=[q_low, 0.5, q_high],
            )
            ppd_low, ppd_high = ppd_quantiles[0][:], ppd_quantiles[2][:]

            #PPD Metrics
            metrics_ppd = evaluate_interval_quality(
                y_true_arr, ppd_low, ppd_high, alpha=alpha
            )

            #MAPIE Conformal interval calculation
            y_pred, y_pis = mapie_model.predict_interval(X_test)
            mapie_low = y_pis[:, 0, 0]
            mapie_high = y_pis[:, 1, 0]
            
            #MAPIE Metrics
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
            #Only used for Synthetic datasets
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
            utils.coverage_plot(len(target_confidence_levels), packs,title=f"{conf_level*100}% Coverage Plot MAPPIE VS. PPD- {dataset_name}")

        #Visualization for the 95% confidence interval MAPPIE
        y_pred_90, y_pis_90 = mapie_model.predict_interval(X_test)
        mapie_low_90 = y_pis_90[:, 0, 0]
        mapie_high_90 = y_pis_90[:, 1, 0]
        intervals_90 = np.column_stack([mapie_low_90, mapie_high_90])

        #Scatter plot of MAPIE intervals
        utils.scatter_plot_coverage(
            y_true=y_true_arr,
            medians=y_pred_90,
            intervals=intervals_90,
            n_samples=100,
            title=f"{conf_level*100}% Coverage Scatter Plot MAPPIE - {dataset_name}",
        )

        #Visualization for the 95% confidence interval TabPFNA
        ppd_90 = regressor.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        ppd_low_90, ppd_med_90, ppd_high_90 = ppd_quantiles[0][:], ppd_quantiles[1][:], ppd_quantiles[2][:]
        intervals_90 = np.column_stack([ppd_low_90, ppd_high_90])

        #Scatter plot of MAPIE intervals
        utils.scatter_plot_coverage(
            y_true=y_true_arr,
            medians=ppd_med_90,
            intervals=intervals_90,
            n_samples=100,
            title=f"{conf_level*100}% Coverage Scatter Plot TabPFN PPD - {dataset_name}",
        )

    # Output summary table
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv('rq1results.csv', mode = 'a')
    print("\n================ FINAL RQ1 SUMMARY TABLE ================")
    print(df_summary.to_string(index=False))
    print("=========================================================\n")

#run_rq1_experiment("hetero",1000)
#run_rq1_experiment("tabarena",1000)