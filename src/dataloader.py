import os, sklearn.datasets, openml
import pandas as pd 
import numpy as np
import random


def openml_data(ID):
    """
    Load a dataset from the OpenML repository via dataset ID.

    Parameters
    ----------
    ID : int
        OpenML dataset ID used to identify the dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing the input variables.

    y : pandas.Series
        Target values of the dataset.

    dataset.name : str
        Name of the dataset as provided by OpenML.
    """
    dataset = openml.datasets.get_dataset(ID)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    print(f" - ID: {ID:<6} | Name: {dataset.name}")

    return X, y, dataset.name

def create_target_noise(y, noise_level = 0.2):
    """Creates noise in given dataset to test the model on aleatoric uncertainty"""
    y_std = y.std() #standard deviation of the target variable
    noise = np.random.normal(loc=0.0, scale=noise_level * y_std, size=len(y)) #Additive White Gaussian Noise
    return y + noise

def create_feature_noise(X_train, noise_level = 0.2):
    """Creates noise in given dataset to test the model on aleatoric uncertainty"""
    for i in X_train.select_dtypes(include=[np.number]).columns: #going over every feature column
            x_std = X_train[i].std() #standard deviation for every feature
            if x_std > 0:
                noise = np.random.normal(0, noise_level * x_std, size=len(X_train)) #Additive White Gaussian Noisee
                X_train[i] = X_train[i] + noise
                
    return X_train

def generate_synthetic_data(type="linear", n_samples=1000, v_range=100, noise=True):
    """Parameters
        ----------
        type : str
            Name of the scenario to generate. Available options are:
    
            - ``"linear"``:
                Generates dataset that represents a linear function.
    
            - ``"parabola"``:
                Generates dataset that represents a parabola function.
    
            - ``"constant"``:
                Generates dataset that represents a constant function.
    
        n_samples : int, default=1000
            Number of samples used for the synthetic datasets.

        v_range : int, default=100
            Range of generated x values [-v_range, v_range].
            
        noise : bool, default=True
            Decides if heterodscactic noise is added or not.
    
        Returns
        -------
        X_df : pandas.DataFrame
            DataFrame with a single column ``"feature_1"`` containing the generated X values.
        y_noisy : pandas.Series
            Series containing the target values with added noise.
        uncertainty_per_point : numpy.ndarray
            1D array containing the ground-truth noise standard deviation for each data point.
        y_true : pandas.Series
            Series containing the true, noise-free target values.
        """
    if noise:
        j = 1
    else:
        j = 0

    X = np.linspace(-v_range,v_range, num=n_samples) #Creates evenly spaced samples based from -100 to 100
    
    #Different kinds of functions
    if type == "linear":
        y =  X
    elif type == "parabola":
        y = (X**2)
    elif type == "constant":
        y = np.full(n_samples, 1)

    y_true= y
    x_min = -v_range
    x_max = v_range
    relative_position = (X - x_min) / (x_max - x_min) #Getting the positioning of each point on a scale of 0-1
    
    #Max and min noise to make the noise grow
    y_true_std = np.std(y_true) if np.std(y_true) > 0 else 1.0
    max_noise_std = y_true_std*10
    min_noise_std = 0.01 * y_true_std 
    
    #Growing noise relative to the postion, the more to right a point is, the higher the noise
    uncertainty_per_point = (min_noise_std + (max_noise_std - min_noise_std) * relative_position) * j
    
    noise = np.random.normal(loc=0.0, scale=uncertainty_per_point)
    
    y_noisy = y_true + noise
    
    X_df = pd.DataFrame(X, columns=["feature_1"])

    return X_df, pd.Series(y_noisy), uncertainty_per_point, pd.Series(y_true)

def generate_scenario_data(scenario_name, n_samples=1000):
    """
    Generate datasets for the different experimental scenarios.

    The function provides synthetic datasets for controlled experiments
    as well as real-world regression datasets from the TabArena benchmark.
    Depending on the selected scenario, the generated data can contain
    no noise, heteroscedastic noise, or regions with missing training data.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario to generate. Available options are:

        - ``"clean"``:
            Generates three synthetic datasets without noise:
            Linear, Parabola, and Constant.

        - ``"hetero"``:
            Generates three synthetic datasets with heteroscedastic
            (input-dependent) noise:
            Linear, Parabola, and Constant.

        - ``"data_gap"``:
            Generates three synthetic datasets in which a region of the
            input space is excluded from the training data. This creates
            a region where epistemic uncertainty can be investigated.

        - ``"tabarena"``:
            Loads selected real-world regression datasets from the
            TabArena benchmark via OpenML. Nominal features are encoded
            as integer values and all features and targets are converted
            to ``float32``.

    n_samples : int, default=1000
        Number of samples used for the synthetic datasets.

    Returns
    -------
    list
        A list containing the datasets corresponding to the selected
        scenario.

        For the ``"clean"`` and ``"hetero"`` scenarios, each dataset
        is returned as:

            X, y, dataset_name

        where ``X`` is a feature DataFrame, ``y`` is the target Series,
        and ``dataset_name`` identifies the synthetic dataset.

        For the ``"data_gap"`` scenario, each dataset is returned as:

            X_train, y_train, X_test, y_test, gap_bounds

        For the ``"tabarena"`` scenario, each dataset is returned as:

            X, y, dataset_name
    """
    dataset_ids = [46904,46907,46917,46923,46928,46931,46934,46942,46949,46954,46953,46961,46964] #13 Tabarena Regression Datasets from the leaderboard
    # -------------------------------------------------------------
    # Scenario 0: No Noise
    # -------------------------------------------------------------
    datasets = []
    if scenario_name == "clean":
            X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
                type="linear", n_samples=n_samples, v_range=2.5, noise=False
            )
            datasets.append((X_df,y_noisy,"Linear"))
            X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
                type="parabola", n_samples=n_samples, v_range=2.5, noise=False
            )
            datasets.append((X_df,y_noisy,"Parabola"))
            X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
                type="constant", n_samples=n_samples, v_range=2.5, noise=False
            )
            datasets.append((X_df,y_noisy, "Constant"))
            return datasets


    # -------------------------------------------------------------
    # Scenario 1: Heteroscedastic Noise (Growing Noise)
    # -------------------------------------------------------------
    if scenario_name == "hetero":
        X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
            type="linear", n_samples=n_samples, v_range=2.5, noise=True
        )
        datasets.append((X_df,y_noisy,"Linear"))
        X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
            type="parabola", n_samples=n_samples, v_range=2.5, noise=True
        )
        datasets.append((X_df,y_noisy,"Parabola"))
        X_df, y_noisy, true_noise_std, y_true = generate_synthetic_data(
            type="constant", n_samples=n_samples, v_range=2.5, noise=True
        )
        datasets.append((X_df,y_noisy, "Constant"))
        return datasets

    # -------------------------------------------------------------
    # Scenario 2: Data Gap (Epistemically Challenging Region)
    # -------------------------------------------------------------
    elif scenario_name == "data_gap":
        X_all = np.linspace(-4,4, num=n_samples) #Creates evenly spaced samples based from -100 to 100
        #Linear
        y_true_all =  X
        gap_bounds = random.randrange(-4, 4, step=2)
        train_mask = (X_all < gap_bounds[0]) | (X_all > gap_bounds[1])
                    
        X_train = pd.DataFrame(X_all[train_mask], columns=["feature_1"])
        y_train = pd.Series(y_true_all[train_mask])

        X_test = pd.DataFrame(X_all, columns=["feature_1"])
        y_test = pd.Series(y_true_all)
        datasets.append((X_train, y_train, X_test, y_test, gap_bounds))
        #Parabole
        y_true_all = (X**2)
        gap_bounds = random.randrange(-4, 4, step=2)
        train_mask = (X_all < gap_bounds[0]) | (X_all > gap_bounds[1])
                    
        X_train = pd.DataFrame(X_all[train_mask], columns=["feature_1"])
        y_train = pd.Series(y_true_all[train_mask])

        X_test = pd.DataFrame(X_all, columns=["feature_1"])
        y_test = pd.Series(y_true_all)
        datasets.append((X_train, y_train, X_test, y_test, gap_bounds))
        #Constant
        y_true_all = np.full(n_samples, 1)
        gap_bounds = random.randrange(-4, 4, step=2)
        train_mask = (X_all < gap_bounds[0]) | (X_all > gap_bounds[1])
                    
        X_train = pd.DataFrame(X_all[train_mask], columns=["feature_1"])
        y_train = pd.Series(y_true_all[train_mask])

        X_test = pd.DataFrame(X_all, columns=["feature_1"])
        y_test = pd.Series(y_true_all)
        datasets.append((X_train, y_train, X_test, y_test, gap_bounds))
        
        return datasets

    # -------------------------------------------------------------
    # Scenario 3: Real Datasets from OpenML / TabArena (e.g., 'openml_531')
    # -------------------------------------------------------------
    elif scenario_name.startswith("tabarena"):
        datasets = []
        for id in [0, 3, 6, 8, 10, 11]: #dataset_ids
            datasets.append((openml_data(dataset_ids[id])))

    # 1. Transform Nominal data into unique numbers (1,2,3,4,...)
        for X ,y ,name in datasets:    
            for col in X.columns:
                if X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype):
                    X[col] = X[col].astype("category").cat.codes

            # 2. Convert all columns into floats
            X = X.astype(np.float32)
            y = y.astype(np.float32)
        return datasets
    
    else:
        raise ValueError(
            f"Invalid scenario_name '{scenario_name}'. "
            f"Choose from: ['hetero', 'data_gap', 'tabarena','clean']"
        )
    