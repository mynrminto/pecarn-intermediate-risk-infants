# Calibrated decision-curve analysis and "per 100 infants" tables for the compact (4-variable) and full (14-variable) logistic models
# Usage: python dca_calibrated.py --csv "/path/to/TBI PUD 10-08-2013.csv" --out exploration
import argparse, numpy as np, pandas as pd, warnings, os, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", default="exploration"); ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
OUT = args.out; SEED = args.seed; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.csv, low_memory=False)
u2 = df[(df["AgeTwoPlus"] == 1) & (df["GCSTotal"] == 15) & (df["AMS"] == 0) & (df["SFxPalp"] == 0)]
inter = u2[(u2["High_impact_InjSev"] == 3) | u2["LocLen"].isin([2, 3, 4]) | u2["HemaLoc"].isin([2, 3]) | (u2["ActNorm"] == 0)]
ct = inter[inter["CTDone"] == 1].copy(); y = ct["PosCT"].astype(int).values
def feats(d):
    X = pd.DataFrame(index=d.index)
    X["AgeInMonth"] = d["AgeInMonth"]; X["HemaLoc_2"] = (d["HemaLoc"] == 2).astype(float).where(d["HemaLoc"].notna())
    X["HemaLoc_3"] = (d["HemaLoc"] == 3).astype(float).where(d["HemaLoc"].notna()); X["HemaSize"] = d["HemaSize"].replace(92, 0)
    X["High_impact_InjSev"] = d["High_impact_InjSev"]
    Fc = list(X.columns)
    X["LocLen"] = d["LocLen"].replace(92, 0); X["ActNorm"] = d["ActNorm"]; X["HASeverity"] = d["HASeverity"].replace(92, 0)
    X["VomitNbr"] = d["VomitNbr"].replace(92, 0); X["FontBulg"] = d["FontBulg"]; X["Clav"] = d["Clav"]; X["NeuroD"] = d["NeuroD"]; X["OSI"] = d["OSI"]
    X["Gender"] = d["Gender"].replace({1: 0, 2: 1}); X["HemaLoc_92"] = (d["HemaLoc"] == 92).astype(float).where(d["HemaLoc"].notna())
    for v in [1, 2, 3]: X[f"SeizOccur_{v}"] = (d["SeizOccur"] == v).astype(float).where(d["SeizOccur"].notna())
    return X, Fc, list(X.columns)
X, Fc, F0 = feats(ct)
def lr(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))  # unweighted -> calibrated probabilities
res = {}
for name, cols in [("compact5_LR", Fc), ("full14_LR", F0)]:
    oofs = []
    for r in range(10):
        oof = np.zeros(len(y))
        for a, v in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            m = lr().fit(X.iloc[a][cols], y[a]); oof[v] = m.predict_proba(X.iloc[v][cols])[:, 1]
        oofs.append(oof)
    p = np.mean(oofs, axis=0); aucs = [roc_auc_score(y, o) for o in oofs]
    # bootstrap CI of AUC on averaged OOF
    rng = np.random.RandomState(SEED); b = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y)); b.append(roc_auc_score(y[i], p[i]))
    res[name] = {"p": p, "auc_mean_repeats": float(np.mean(aucs)), "auc_boot_ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "brier": float(brier_score_loss(y, p)),
                 "calibration_mean_pred": float(p.mean()), "prevalence": float(y.mean())}
    print(name, "CV AUC %.3f (boot 95%% CI %.3f-%.3f) Brier %.4f mean pred %.3f prev %.3f" % (res[name]["auc_mean_repeats"], *res[name]["auc_boot_ci"], res[name]["brier"], p.mean(), y.mean()))
prev = y.mean(); pts = np.round(np.arange(0.01, 0.16, 0.005), 3)
def nb(y, p, pt):
    pred = p >= pt; n = len(y); return (pred & (y == 1)).sum() / n - (pred & (y == 0)).sum() / n * pt / (1 - pt)
rows = []
for pt in pts:
    row = {"threshold_prob": pt, "NB_CT_all": prev - (1 - prev) * pt / (1 - pt)}
    for name in res:
        p = res[name]["p"]; pred = p >= pt
        row[f"NB_{name}"] = nb(y, p, pt); row[f"{name}_CT_avoided_per100"] = 100 * (~pred).mean()
        row[f"{name}_TBI_missed_per100"] = 100 * (~pred & (y == 1)).sum() / len(y); row[f"{name}_sens"] = pred[y == 1].mean()
    rows.append(row)
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, "dca_calibrated.csv"), index=False)
print(tab[tab["threshold_prob"].isin([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1])].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pts, tab["NB_CT_all"], "k-", lw=1.5, label="CT for all (current practice)"); ax.plot(pts, [0] * len(pts), "k--", lw=1, label="CT for none")
for name, lab in [("compact5_LR", "4-variable logistic model (age, hematoma site/size, mechanism)"), ("full14_LR", "14-variable logistic model")]:
    ax.plot(pts, tab[f"NB_{name}"], lw=1.8, label=lab)
ax.set_xlabel("Threshold probability of TBI on CT at which CT would be performed"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.005, prev + 0.005)
ax.set_title("Decision curve (calibrated out-of-fold probabilities, 10 x 5-fold CV)"); ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "decision_curve_calibrated.png"), dpi=200)
# coefficients of the compact model on the full cohort (for a possible nomogram / score)
m = lr().fit(X[Fc], y); sc = m.named_steps["standardscaler"]; coef = m.named_steps["logisticregression"].coef_[0] / sc.scale_
b0 = m.named_steps["logisticregression"].intercept_[0] - np.sum(coef * sc.mean_)
print("compact model (raw scale) intercept %.3f" % b0, {c: round(float(v), 3) for c, v in zip(Fc, coef)}, "ORs:", {c: round(float(np.exp(v)), 2) for c, v in zip(Fc, coef)})
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "p"} for k, v in res.items()} | {"compact_coef_raw": {c: float(v) for c, v in zip(Fc, coef)}, "compact_intercept": float(b0)}, open(os.path.join(OUT, "dca_calibrated_summary.json"), "w"), indent=2)
