# Risk stratification of infants younger than 2 years in the PECARN intermediate-risk group — analysis code

This repository contains the complete, reproducible analysis for the manuscript
*"Risk stratification of infants younger than 2 years in the PECARN intermediate-risk group: a secondary analysis"* (manuscript in preparation).

## Data

The analysis uses the **PECARN TBI Public Use Dataset, version 1.0** (`TBI PUD 10-08-2013.csv`), available from the
PECARN Data Coordinating Center (https://pecarn.org/datasets/) under a Research Data Use Agreement. The dataset is **not**
redistributed here.

## Scripts

| Script | Purpose | Output |
|---|---|---|
| `analysis_v3.py` | **Main analysis for the manuscript.** Cohort selection (Cohort A: all intermediate-risk infants; Cohort B: imaged subset), Table 1, risk tables with exact 95% CIs (Table 2), model comparison by 10 × 5-fold cross-validation (Table S1), coefficient-derived point score (Table 3), safety-oriented criteria and exhaustive rule search (Table 4), management pathway, decision curves, Figures 1–4 | `results_v3/` |
| `model_exploration.py` | Systematic comparison of feature sets, models and training pools (repeated cross-validation, decision curves) that motivated the final design | `exploration/` |
| `dca_calibrated.py` | Calibrated decision-curve analysis and outcomes per 100 infants for the compact and full logistic models | `exploration/` |

## How to run

```bash
pip install -r requirements.txt   # tested with Python 3.12
# main analysis (about 3 minutes on a laptop)
python analysis_v3.py --csv "/path/to/TBI PUD 10-08-2013.csv" --out results_v3
# supplementary analyses (not required to reproduce the manuscript)
python model_exploration.py --csv "/path/to/TBI PUD 10-08-2013.csv" --out exploration
python dca_calibrated.py --csv "/path/to/TBI PUD 10-08-2013.csv" --out exploration
```

Random seeds are fixed (`--seed 42`); repeated runs produce identical outputs.

## Notes

- The "4-variable model" of the manuscript (age in months, hematoma location, hematoma size, mechanism severity) has five
  model terms because hematoma location enters as two indicator terms. It is therefore named `LR5` / `F5` /
  `compact5_LR` in the code and output files (`lr5_coefficients.csv`, etc.).
- `analysis_v3.py` excludes one infant with missing ciTBI status (Cohort A n = 3,470; Cohort B n = 1,348). The two
  supplementary scripts were written earlier and keep that infant (imaged intermediate-risk infants n = 1,349).
- Output file names keep their original numbering and do not follow the table numbers of the manuscript: `table3_model_comparison.csv` is Table S1, `table5_score_levels.csv` is Table 3, `table_safety_criteria.csv` is Table 4, and `table4_decision_curve.csv` underlies Figure 3c.
- `results_v3/` and `exploration/` contain aggregate outputs only (tables, figures, logs). No individual-level data are included:
  the case-level listings that `analysis_v3.py` writes to `japan_context.json` and `log.txt` (descriptions of individual ciTBI
  cases) have been removed from the copies distributed here, in accordance with the PECARN data use agreement. Please do
  not redistribute those listings when you run the code on your own copy of the dataset.

## Manuscript generation

`build_manuscript_v3.js` (Node.js, `docx` package; kept with the manuscript files, not in this repository) renders the manuscript directly from the files in `results_v3/`,
so that every number in the text, tables and figure legends is traceable to the analysis output:

```bash
node build_manuscript_v3.js Manuscript.docx analysis/results_v3
```

## Use of generative AI

The analysis and figure-generation code was written with the assistance of generative AI (Claude, Anthropic) under the authors' direction. The authors reviewed the code and verified the results against the analysis output.

## Citation

If you use this code, please cite the manuscript and acknowledge PECARN: "The PECARN was funded by the HRSA/MCHB/EMSC."

## License

MIT, see `LICENSE` (code only; the PECARN dataset is governed by its own data use agreement).
