import os
import argparse
from rq1 import run_rq1_experiment
from rq2 import rq2_experiment
from rq3 import run_rq3_experiment
from rq4 import run_rq4_1_experiment, rq4_2_experiment, rq4_inference

MODELS = ["TabPFN_v2", "TabPFN_v2.5", "TabPFN_v3", "TabICL"]

def main():
    parser = argparse.ArgumentParser(
        description="Bachelor Thesis Benchmark: Tabular Foundation Models & Uncertainty Quantification"
    )
    parser.add_argument(
        "--rq", 
        type=str, 
        choices=["1", "2", "3", "4", "all"], 
        default="all",
        help="Select the Research Question to execute (default: all, Warning: all does not contain rq4, because plot paths in 'mse_plot_multi_folds' and "
              "'plot_sparsification_multi_sets' are hardcoded and have to be changed )."
    )
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="clean", 
        help="Dataset scenario name (e.g., 'clean', 'tabarena')."
    )

    args = parser.parse_args()

    # RQ1 Pipeline
    if args.rq in ["1", "all"]:
        print("\n==================================================")
        print(">>> Starting RQ1 Experiment...")
        print("==================================================")
        run_rq1_experiment(scenario=args.scenario)

    # RQ2 Pipeline
    if args.rq in ["2", "all"]:
        print("\n==================================================")
        print(">>> Starting RQ2 Experiment...")
        print("==================================================")
        rq2_experiment(scenario=args.scenario)

    # RQ3 Pipeline
    if args.rq in ["3", "all"]:
        print("\n==================================================")
        print(">>> Starting RQ3 Experiment...")
        print("==================================================")
        run_rq3_experiment(scenario=args.scenario)

    # RQ4 Pipeline
    if args.rq in ["4"]:
        print("\n==================================================")
        print(">>> Starting RQ4 Complete Experimental Pipeline...")
        print("==================================================")
        print("[NOTE] Ensure plot paths in 'mse_plot_multi_folds' and "
              "'plot_sparsification_multi_sets' are redirected to RQ4 folders.\n")
        
        # 1. Model evaluation loop for RQ4 experiments
        for model in MODELS:
            print(f"---> Starting RQ4 Experimentation with model: {model}")
            run_rq4_1_experiment(model=model)
            rq4_2_experiment(model=model)
        
        # 2. Inference latency benchmark across all models
            print("\n---> Starting RQ4 Inference Benchmark...")
            rq4_inference()

if __name__ == "__main__":
    main()