Bachelor Thesis: Benchmarking Tabular Foundation Models & Uncertainty Quantification

This repository contains the official source code, evaluation tools, and experimental execution pipelines for the Bachelor Thesis. The benchmark evaluates state-of-the-art **Tabular Foundation Models** (**TabPFN v2, v2.5, v3** and **TabICL**) across prediction accuracy, uncertainty quantification (Conformal Prediction vs. Martingale Posteriors), robustness under noise/scaling regimes, and inference efficiency.

---

## 📁 Repository Structure

```text

main.py                 # Central CLI entry point for executing research questions
dataloader.py           # Dataset generation and benchmark scenario loading
uq_Methods.py           # Martingale Posterior uncertainty decomposition routines
utils.py                # Quantile adapters, evaluation metrics, and plot utilities
rq1.py                  # RQ1 experimental pipeline execution
rq2.py                  # RQ2 experimental pipeline execution
rq3.py                  # RQ3 experimental pipeline (Sample Size & Noise Scaling)
rq4.py                  # RQ4 experimental pipeline (Model Comparisons & Latency)
requirements.txt        # Python dependency specifications
README.md               # Repository setup and execution guide



Important Code Modifications & Prerequisites
--------------------------------------------
Before launching the experimental pipelines, ensure the following code modifications and file dependencies are configured:

1. TabICL Code Modification (quantiles output)
To integrate TabICL seamlessly into the evaluation pipeline shared with TabPFN, two source code adaptations were made to TabICL:

Quantile Ordering Alignment: The array returned by predict() when using quantile predictions was reversed to match the dimension layout and quantile ordering expected by TabPFN.

Parameter Interface: The function signature was updated to accept quantiles (replacing TabICL's original internal parameter name) to support unified pipeline calls.

2. Output Paths for RQ4 Plot Utilities
During RQ4 execution, plotting routines originally designed for RQ2 (mse_plot_multi_folds and plot_sparsification_multi_sets) are reused.

Path Overwrite Warning: By default, these routines write output figures to the original RQ2 results directory (e.g., rq2_results/ or plots/rq2/).

Required Action: Update the destination directory argument (e.g., set save_dir="rq4_results/plots") in these function calls inside rq4.py prior to running RQ4 to prevent overwriting RQ2 plot artifacts.

3. Model Checkpoint Placement
Ensure that the required TabPFN model weights are located in the project root directory:

tabpfn-v2-regressor.ckpt

tabpfn-v2.5-regressor-v2.5_default.ckpt

⚙️ Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/Fieo-Online/Uncertainty-Quantification-Methods-for-TabPFN](https://github.com/Fieo-Online/Uncertainty-Quantification-Methods-for-TabPFN)
cd Uncertainty-Quantification-Methods-for-TabPFN
Activate your environment (e.g., Conda/Miniconda):

Bash
conda activate UQ_Methods_TabPFN
Install dependencies:

Bash
pip install -r requirements.txt
🚀 Running Experiments (main.py)
All experimental pipelines can be run directly via the central main.py interface.

Running Specific Research Questions
Run RQ3 (Sample Size & Noise Scaling Regimes):

Bash
python main.py --rq 3
Run RQ4 (Complete Model Benchmarking & Inference Timing):
Executing RQ4 automatically iterates through all models (TabPFN_v2, TabPFN_v2.5, TabPFN_v3, and TabICL) across both sub-experiments (run_rq4_1_experiment and rq4_2_experiment) before evaluating inference latency (rq4_inference):

Bash
python main.py --rq 4
Run All Research Questions Sequentially:

Bash
python main.py --rq all

📊 Output Files & Artifacts
The execution outputs are saved automatically in structured CSV formats and plot figures:

rq1results.csv: Contains performance metrics like conformal coverage rates and interval widths

rq3results.csv: Contains performance metrics, conformal coverage rates, and UQ decay under varying training sample sizes and injected noise levels.

rq4_1results.csv: Contains performance metrics like conformal coverage rates and interval widths for each tested model in RQ4

rq4_inference_results.csv: Benchmarking results containing mean, median, and standard deviation of inference execution latency per model and dataset.

Plot Figures (.png): Output figures generated during evaluation are saved in their corresponding folder paths (results/rq1, results/rq2, results/rq3, or results/rq4/).