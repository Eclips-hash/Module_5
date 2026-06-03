# Does ICU-type stratification improve mortality prediction? Comparing specialist and global XGBoost models with APACHE-IVa on GOSSIS (WiDS 2020\)

**Module 5 coursework, MSt Healthcare Data Science, University of Cambridge.** **Author:** Hanna Hong.

This repository accompanies my Module 5 report. It contains the full Python pipeline used to develop and evaluate the models described in the report, along with all numerical outputs and figures that the report references.

The Module 5 submission has three components, of which this repository is one:

1. **Coversheet**  
2. **Report**   
3. **GitHub repository** (https://github.com/Eclips-hash/Module\_5)

# Abstract
### Background.
APACHE-IVa is a frozen 2006 logistic regression fit to US-only data with no unit-specific adaptation. Modern ML on the same first-24-hour features admits two competing strategies: a global model that learns unit structure through ICU type as a feature, or specialist models with a separate decision boundary per ICU at the cost of an eight-fold smaller training sample. Whether either beats APACHE-IVa on a large multinational cohort, and whether specialisation pays off, has not been settled on the GOSSIS dataset.
### Methods. 
On the WiDS 2020 / GOSSIS dataset (91,713 ICU stays, 8.6% mortality), I compared APACHE-IVa, class-weighted logistic regression, a tuned isotonically calibrated global XGBoost, and eight specialist XGBoosts. AUC-PR with paired-bootstrap CIs (n = 16,288) was primary, supplemented by decision-curve analysis, threshold metrics, SHAP, and a subgroup audit.
Results. The global XGBoost achieved AUC-PR 0.585 (95% CI 0.560–0.611), exceeding APACHE-IVa (0.456) and specialists (0.528). Isotonic calibration halved Brier from 0.102 to 0.053 and delivered the highest net benefit across clinically relevant thresholds (NNE 3.78 vs APACHE-IVa 4.59). No specialist outperformed the global model.
### Conclusion. 
A single calibrated global XGBoost outperformed both APACHE-IVa and specialist models. External validation on a non-US cohort is needed.

## Quick start for assessors

To regenerate every number, table, and figure in the report:

1. Download `training_v2.csv` from Kaggle (see [Dataset access](#dataset-access) below) and place it where the notebook expects.  
2. Open `ICU_mortality_pipeline_v3_revised.ipynb` in Google Colab.  
3. Update `INPUT_PATH` in Cell 1 to point at the CSV.  
4. Run all cells top to bottom. Expected runtime: 25–35 minutes on the standard Colab CPU runtime.  
5. All outputs (CSVs, PNGs, `results_summary.json`) are written to the directory set by `OUTPUT_DIR` in Cell 1\. The pre-generated outputs from my own run are committed to the `results/` folder so the report numbers can be cross-checked without rerunning.

Seed is fixed at 42 throughout, so the run is deterministic given the same `training_v2.csv` and the package versions in `requirements.txt` (set `n_jobs=1` in Cell 7 for byte-identical reproduction; the default `n_jobs=-1` introduces small parallel non-determinism in which configuration `RandomizedSearchCV` ultimately selects, but does not affect the headline findings).

---

## Dataset access

The dataset is **not committed to this repository**. The WiDS Datathon 2020 / GOSSIS-1 data are released under a use-only licence that does not permit redistribution. To reproduce the analysis, each user must obtain their own copy via the external link below:

**Dataset link:** [https://www.kaggle.com/c/widsdatathon2020/data](https://www.kaggle.com/c/widsdatathon2020/data)

Access steps:

1. Sign in to Kaggle and accept the WiDS Datathon 2020 competition terms.  
2. Download `training_v2.csv` (\~86 MB, 91,713 rows, 186 columns).  
3. Place the file in `data/training_v2.csv` (relative to the repo root), or update `INPUT_PATH` in Cell 1 of the notebook to wherever you save it. If running in Colab, common options are: upload to the session at `/content/training_v2.csv`, or mount Google Drive and read from there.

The labelled training file is the only data file used. The competition's unlabelled set is not used because its labels are not publicly released.

**Dataset citation:** Lee M, Raffa J, Ghassemi M, Pollard T, Hooper S, Malik V, Mark RG, Badawi O. *WiDS (Women in Data Science) Datathon 2020: ICU Mortality Prediction.* Kaggle (2020). The underlying GOSSIS-1 cohort spans over 200 hospitals across six countries.

---

## Software requirements

The pipeline was developed and tested on **Google Colab with Python 3.11**.   
Key packages and what they're used for:

| Package | Version | Purpose |
| :---- | :---- | :---- |
| `python` | 3.11 | base interpreter |
| `numpy` | 1.26.4 | numerical |
| `pandas` | 2.2.2 | data handling |
| `scipy` | 1.13.1 | helper functions |
| `scikit-learn` | 1.5.1 | LogReg baseline, `RandomizedSearchCV`, `IsotonicRegression`, metrics |
| `xgboost` | 2.1.0 | global \+ specialist XGBoosts (≥ 2.0 required) |
| `shap` | 0.46.0 | feature attribution (`TreeExplainer`) |
| `matplotlib` | 3.8.4 | plotting |

**XGBoost 2.0 or later is mandatory.** The pipeline uses the constructor-level `early_stopping_rounds` argument that moved to the constructor in XGBoost 2.0. On older versions the code raises `TypeError` at fit time.

If you run the notebook on Colab, the first code cell prints the active versions of the key packages so you can confirm they match this file before interpreting any results.  
---

## How to run

### Google Colab (matches the development environment)

1. Open the notebook directly in Colab: `File → Upload Notebook`, or open from GitHub using Colab's URL opener.  
2. In Cell 1, set `INPUT_PATH` to wherever you placed `training_v2.csv`. If you uploaded it to the Colab session this is usually `/content/training_v2.csv`; if you mounted Drive it will be a path under `/content/drive/MyDrive/...`.  
3. Use `Runtime → Run all`. On standard Colab CPU, expect 25–35 minutes total. The `RandomizedSearchCV` over the global XGBoost (Cell 7\) is the slowest single step at \~10–15 minutes.  
4. Outputs are written into `OUTPUT_DIR` (default `/content/outputs/`).

---

## What each cell does

The notebook is organised into 20 numbered cells. Each has a short markdown narrative immediately above the code explaining the methodological choice. The "Maps to report" column shows where each cell's output is used in the report.

| Cell | Purpose | Maps to report |
| :---- | :---- | :---- |
| 1 | Setup, imports, seed (42), `OUTPUT_DIR` | §2.5 |
| 2 | CSV sanity checks; recode APACHE-IVa missingness; mortality by ICU type | §3.1 |
| 3 | Feature engineering: range, deterioration, comorbidity count, missingness flags | §2.2 |
| 4 | Stratified 80:20 train/test split (joint outcome × ICU type) | §2.3 |
| 5 | APACHE-IVa benchmark | §3.2 |
| 6 | Logistic regression baseline (class-weighted) | §3.2 |
| 7 | Global XGBoost with `RandomizedSearchCV` | §2.3, §3.2 |
| 8 | Specialist XGBoosts with minimum-iteration safeguard | §2.3, §3.5 |
| 9 | Probability calibration of the global XGBoost (isotonic) | §2.4, §3.2 |
| 10 | XGBoost seed stability check | §3.8 |
| 11 | Paired bootstrap 95% CIs across all five models | §2.5, §3.2 |
| 12 | Headline tables → CSV outputs | Tables 1, 1b, 2 |
| 13 | ROC, precision–recall, and calibration plots | Figure 1 |
| 14 | Threshold metrics at sensitivity ≈ 0.85 | §3.3, Table 2 |
| 15 | Decision-curve analysis | §3.4, Figure 2 |
| 16 | Per-ICU forest plot | §3.5, Figure 3 |
| 17 | SHAP attribution (TreeExplainer on the calibrated model) | §3.6, Figure 4 |
| 18 | Subgroup-fairness audit (ethnicity, gender) | §3.7, Table 4 |
| 19 | Multi-seed logistic regression stability | §3.8 |
| 20 | Save all numerical results to a single `results_summary.json` | — |

---

## Infrastructure

The pipeline was developed and finalised on **Google Colab using the free CPU runtime** (typically Intel Xeon with \~12 GB RAM). I did not use Colab Pro or any cloud compute.

The pipeline will also run on:

- **Local CPU**, any modern laptop with ≥ 8 GB RAM; expect 30–45 minutes total runtime.  
- **Local GPU**: not faster for this pipeline. XGBoost is CPU-bound at this dataset size on default settings, and `shap.TreeExplainer` does not use GPU.

Memory usage peaks at around 4–5 GB during the XGBoost hyperparameter search (Cell 7), well within Colab's 12 GB limit.

A note on parallelism: Cell 7 uses `n_jobs=-1` in `RandomizedSearchCV` to parallelise across CPU cores. This introduces tiny non-determinism in which configuration is selected (parallel workers can finish in different orders across runs). For byte-identical reproduction, change to `n_jobs=1`. Across multiple runs I confirmed the chosen best configuration was always within the same neighbourhood of the search grid, and the headline AUC-PR, Brier, and NNE numbers did not change in any clinically meaningful way.

---

## Reproducibility notes

- Random seed is fixed at 42 across NumPy, scikit-learn, and XGBoost.  
- The train/test split, bootstrap resampling, CV folds, and XGBoost's internal randomness all derive from the same seed.  
- Stability is evaluated explicitly: Cell 10 (XGBoost) and Cell 19 (logistic regression) re-run the pipeline under seeds 0, 7, and 42 and report the AUC-ROC and AUC-PR range. Both ranges sit within the paired-bootstrap CI widths reported in Table 1, and rank ordering of models is preserved across all three seeds. This is §3.8 of the report.  
- **One catch in the data load**: the default `pandas.read_csv` C parser silently truncated `training_v2.csv` to \~62,000 rows on my development machine, dropping about 30,000. The pipeline uses `engine='python'` to avoid this, and Cell 2 asserts on the expected row count (91,713) so any future truncation crashes loudly rather than silently producing wrong numbers.  
- **One catch in early stopping**: in my first run, the CCU-CTICU specialist's early stopping fired at iteration 5, which is clearly a misfire on a small specialist with a small validation slice. Cell 8 includes a minimum-iteration floor (`MIN_ITER_FLOOR = 20`) that detects this and refits the model without early stopping at 200 trees. This is reported in §4.3 of the report as a limitation.

---

## Mapping to the assessment requirements

For the assessor's convenience, here is where each report-content item from the Module 5 descriptor is addressed:

**Part 1 — Background, rationale, and model development (60%):**

| Descriptor item | Location |
| :---- | :---- |
| Healthcare problem and research question | Report §1.1, §1.4 |
| Justification for the choice of dataset | Report §1.4, §2.1 |
| Background on ML relevance to the problem | Report §1.1, §1.2 |
| Data preparation and preprocessing | Report §2.1, §2.2; notebook Cells 2–3 |
| Related ML approaches | Report §1.2 |
| Feature engineering | Report §2.2; notebook Cell 3 |
| Justification for the choice of algorithms | Report §2.3 |
| Model selection strategy | Report §2.3, §2.4 |
| Parameter tuning / optimisation | Report §2.3; notebook Cell 7 |
| Modelling pipeline rationale | Report §2.3–§2.5 |
| Stability across seeds or folds | Report §3.8; notebook Cells 10, 19 |

**Part 2 — Model evaluation and interpretation (40%):**

| Descriptor item | Location |
| :---- | :---- |
| Performance metrics | Report §2.5, §3.2 |
| Model comparison | Report §3.2, §3.5; Tables 1–3; Figure 3 |
| Strengths and limitations | Report §4.3 |
| Healthcare-context interpretation | Report §4.1, §4.2 |
| Generalisability, bias, fairness, transparency, explainability | Report §3.6, §3.7, §4.4 |
| Trade-offs (interpretability, computational cost) | Report §4.1, §4.3 |
| Ethical reflection | Report §4.4 |

---

## Data licence reminder

The WiDS Datathon 2020 / GOSSIS-1 data are under a use-only licence. **Do not commit `training_v2.csv` (or any derivative containing patient-level data) to a public repository.** The `.gitignore` excludes `data/` and any stray CSVs from version control.  
The trained-model artefacts and predictions also fall under the original GOSSIS licence terms; they should not be used clinically without the validation and recalibration steps described in §4.5 of the report.

---

## Contact

Hanna Hong, hnh27@cam.ac.uk 


