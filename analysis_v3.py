#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis for manuscript v3: "Risk stratification within the PECARN intermediate-risk group in infants < 2 years"

Cohort A  all PECARN intermediate-risk infants < 2 years (GCS 15, no AMS, no palpable skull fracture, >= 1 intermediate
          finding), regardless of imaging (n = 3,470 after excluding 1 infant with missing ciTBI status). Outcome: clinically important TBI (ciTBI, ascertained for all by
          follow-up) -> risk tables with exact 95% CIs, exploratory multivariable logistic regression.
Cohort B  the imaged subset (n = 1,348). Outcome: TBI on CT. 4-variable logistic model (age in months, hematoma location,
          hematoma size, mechanism severity; 5 model terms, hence the internal name "LR5") and a simple point score; 10 x 5-fold repeated CV -> AUC (bootstrap CI),
          calibration, decision curve, outcomes per 100 infants at candidate thresholds; comparison with the 14-variable
          logistic model, random forest, LightGBM, and an extended 51-variable logistic model.
Bridge    the point score (derived in Cohort B) applied to Cohort A -> ciTBI risk per score level with less verification bias.

Usage: python analysis_v3.py --csv "/path/to/TBI PUD 10-08-2013.csv" --out results_v3
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", default="results_v3")
ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
OUT = args.out; SEED = args.seed; os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "log.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
def cp_ci(k, n):
    if n == 0: return (np.nan, np.nan)
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return (100 * lo, 100 * hi)
def fmt_risk(k, n):
    lo, hi = cp_ci(k, n); return f"{k}/{n:,} ({100*k/n:.2f}%; {lo:.2f}–{hi:.2f})" if n else "–"

# ----------------------------------------------------------------------------- cohorts
df = pd.read_csv(args.csv, low_memory=False)
u2 = df[df["AgeTwoPlus"] == 1]
s1 = u2[(u2["GCSTotal"] == 15) & (u2["AMS"] == 0) & (u2["SFxPalp"] == 0)]
A = s1[(s1["High_impact_InjSev"] == 3) | s1["LocLen"].isin([2, 3, 4]) | s1["HemaLoc"].isin([2, 3]) | (s1["ActNorm"] == 0)].copy()
flow = {"total": len(df), "age_lt2": len(u2), "gcs15_noAMS_noSFx": len(s1), "intermediate": len(A),
        "excluded_high_or_missing": len(u2) - len(s1), "excluded_very_low": len(s1) - len(A),
        "ciTBI_missing": int(A["PosIntFinal"].isnull().sum())}
A = A[A["PosIntFinal"].notna()].copy(); A["ciTBI"] = A["PosIntFinal"].astype(int)
flow["cohortA"] = len(A); flow["ciTBI"] = int(A["ciTBI"].sum()); flow["neurosurgery"] = int((A["Neurosurgery"] == 1).sum())
flow["death"] = int((A["DeathTBI"] == 1).sum()); flow["intub24"] = int((A["Intub24Head"] == 1).sum()); flow["hosp2"] = int((A["HospHeadPosCT"] == 1).sum())
B = A[A["CTDone"] == 1].copy(); flow["cohortB"] = len(B); flow["TBIonCT"] = int((B["PosCT"] == 1).sum()); flow["ciTBI_in_B"] = int(B["ciTBI"].sum())
flow["ciTBI_not_imaged"] = int(A[A["CTDone"] != 1]["ciTBI"].sum()); flow["not_imaged"] = int((A["CTDone"] != 1).sum())
json.dump({k: int(v) for k, v in flow.items()}, open(os.path.join(OUT, "flow.json"), "w"), indent=2); P("flow", flow)

# derived categories
def derive(d):
    d = d.copy()
    d["age_grp"] = pd.cut(d["AgeInMonth"], [-1, 2, 5, 11, 23], labels=["0–2 months", "3–5 months", "6–11 months", "12–23 months"])
    d["hema_cat"] = np.select([d["HemaLoc"] == 92, d["HemaLoc"] == 1, d["HemaLoc"].isin([2, 3]) & (d["HemaSize"].isin([1, 2])),
                               d["HemaLoc"].isin([2, 3]) & (d["HemaSize"] == 3)],
                              ["No hematoma", "Frontal hematoma", "Non-frontal hematoma, ≤3 cm", "Non-frontal hematoma, >3 cm"], "Hematoma, location/size unknown")
    d["severe_mech"] = (d["High_impact_InjSev"] == 3).astype(int); d["loc5"] = d["LocLen"].isin([2, 3, 4]).astype(int)
    d["not_normal"] = (d["ActNorm"] == 0).astype(int); d["nonfrontal"] = d["HemaLoc"].isin([2, 3]).astype(int)
    d["n_find"] = d["severe_mech"] + d["loc5"] + d["nonfrontal"] + d["not_normal"]
    d["n_find_cat"] = np.where(d["n_find"] >= 2, "≥2 findings", "1 finding")
    return d
A = derive(A); B = derive(B)

# ----------------------------------------------------------------------------- point score derived from regression coefficients (cohort B)
# Logistic regression for TBI on CT with categorical predictors; points = coefficient / smallest coefficient >= 0.15, rounded (Framingham-type)
def X5cat(d):
    X = pd.DataFrame(index=d.index)
    X["age_0_2"] = (d["AgeInMonth"] < 3).astype(int); X["age_3_5"] = ((d["AgeInMonth"] >= 3) & (d["AgeInMonth"] < 6)).astype(int); X["age_6_11"] = ((d["AgeInMonth"] >= 6) & (d["AgeInMonth"] < 12)).astype(int)
    X["occipital"] = (d["HemaLoc"] == 2).astype(int); X["parietal_temporal"] = (d["HemaLoc"] == 3).astype(int)
    X["size_1_3cm"] = (d["HemaSize"] == 2).astype(int); X["size_gt3cm"] = (d["HemaSize"] == 3).astype(int)
    X["mech_moderate"] = (d["High_impact_InjSev"] == 2).astype(int); X["mech_severe"] = (d["High_impact_InjSev"] == 3).astype(int)
    return X.fillna(0)
Xcat = X5cat(B); ycat = (B["PosCT"] == 1).astype(int).values
lr_cat = LogisticRegression(C=1.0, max_iter=5000).fit(Xcat, ycat)
coef_cat = dict(zip(Xcat.columns, lr_cat.coef_[0])); unit = min(v for v in coef_cat.values() if v >= 0.15)
POINTS = {k: int(round(v / unit)) for k, v in coef_cat.items()}
def score_of(d): X = X5cat(d); return sum(X[k] * p for k, p in POINTS.items()).astype(int)
A["score"] = score_of(A); B["score"] = score_of(B)
# cross-validated AUC of the score when derived within folds (to quantify optimism)
oof_sc = np.zeros(len(B))
for r in range(args.repeats):
    for a, v in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xcat, ycat):
        mm = LogisticRegression(C=1.0, max_iter=5000).fit(Xcat.iloc[a], ycat[a]); cc = dict(zip(Xcat.columns, mm.coef_[0])); uu = min(x for x in cc.values() if x >= 0.15)
        pp = {k: int(round(x / uu)) for k, x in cc.items()}; oof_sc[v] += sum(Xcat.iloc[v][k] * q for k, q in pp.items()) / args.repeats
json.dump({"coefficients": {k: float(v) for k, v in coef_cat.items()}, "unit": float(unit), "points": POINTS, "max_score": int(max(POINTS["age_0_2"], POINTS["age_3_5"], POINTS["age_6_11"], 0) + max(POINTS["occipital"], POINTS["parietal_temporal"], 0) + max(POINTS["size_1_3cm"], POINTS["size_gt3cm"], 0) + max(POINTS["mech_moderate"], POINTS["mech_severe"], 0)),
           "auc_score_B_apparent": float(roc_auc_score(ycat, B["score"])), "auc_score_B_cv_derived": float(roc_auc_score(ycat, oof_sc)), "auc_score_A_ciTBI": float(roc_auc_score(A["ciTBI"], A["score"]))},
          open(os.path.join(OUT, "score_definition.json"), "w"), indent=2)
P("score points:", POINTS, "apparent AUC B %.3f, CV-derived %.3f, ciTBI AUC A %.3f" % (roc_auc_score(ycat, B["score"]), roc_auc_score(ycat, oof_sc), roc_auc_score(A["ciTBI"], A["score"])))

# ----------------------------------------------------------------------------- Table 1: characteristics of cohort A by imaging
def n_pct(mask, n): k = int(mask.sum()); return f"{k:,} ({100*k/n:.1f})"
rows = []
grp = {"All": A, "CT performed": A[A["CTDone"] == 1], "No CT": A[A["CTDone"] != 1]}
def add(label, fn): rows.append([label] + [fn(g) for g in grp.values()])
add("n", lambda g: f"{len(g):,}")
add("Age, months, median (IQR)", lambda g: f"{g['AgeInMonth'].median():.0f} ({g['AgeInMonth'].quantile(.25):.0f}–{g['AgeInMonth'].quantile(.75):.0f})")
for lab in ["0–2 months", "3–5 months", "6–11 months", "12–23 months"]: add("  " + lab, lambda g, lab=lab: n_pct(g["age_grp"] == lab, len(g)))
add("Male sex", lambda g: n_pct(g["Gender"] == 1, len(g)))
add("Severe injury mechanism", lambda g: n_pct(g["severe_mech"] == 1, len(g)))
add("Loss of consciousness ≥5 s", lambda g: n_pct(g["loc5"] == 1, len(g)))
add("Not acting normally per parent", lambda g: n_pct(g["not_normal"] == 1, len(g)))
for lab in ["No hematoma", "Frontal hematoma", "Non-frontal hematoma, ≤3 cm", "Non-frontal hematoma, >3 cm", "Hematoma, location/size unknown"]:
    add("  " + lab, lambda g, lab=lab: n_pct(g["hema_cat"] == lab, len(g)))
add("Vomiting", lambda g: n_pct(g["Vomit"] == 1, len(g)))
add("Post-traumatic seizure", lambda g: n_pct(g["Seiz"] == 1, len(g)))
add("Observed in the ED before CT decision", lambda g: n_pct(g["Observed"] == 1, len(g)))
add("Number of intermediate-risk findings ≥2", lambda g: n_pct(g["n_find"] >= 2, len(g)))
add("Clinically important TBI", lambda g: n_pct(g["ciTBI"] == 1, len(g)))
add("  Neurosurgery", lambda g: n_pct(g["Neurosurgery"] == 1, len(g)))
add("  Hospital admission ≥2 nights with TBI on CT", lambda g: n_pct(g["HospHeadPosCT"] == 1, len(g)))
add("TBI on CT (among imaged)", lambda g: n_pct(g["PosCT"] == 1, len(g)) if (g["CTDone"] == 1).all() else "–")
pd.DataFrame(rows, columns=["Characteristic", "All", "CT performed", "No CT"]).to_csv(os.path.join(OUT, "table1.csv"), index=False)

# ----------------------------------------------------------------------------- Table 2: ciTBI risk by subgroup (cohort A) + TBI on CT among imaged
def risk_rows(by, labels=None):
    out = []
    keys = labels if labels is not None else sorted(A[by].dropna().unique())
    for k in keys:
        a = A[A[by] == k]; b = a[a["CTDone"] == 1]
        out.append({"group": by, "level": str(k), "n": len(a), "ciTBI": int(a["ciTBI"].sum()), "ciTBI_pct": 100 * a["ciTBI"].mean(),
                    "ci_lo": cp_ci(int(a["ciTBI"].sum()), len(a))[0], "ci_hi": cp_ci(int(a["ciTBI"].sum()), len(a))[1],
                    "CT_rate_pct": 100 * (a["CTDone"] == 1).mean(), "imaged_n": len(b), "TBIonCT": int((b["PosCT"] == 1).sum()),
                    "TBIonCT_pct": 100 * (b["PosCT"] == 1).mean() if len(b) else np.nan})
    return out
t2 = []
t2 += [{"group": "Overall", "level": "All intermediate-risk infants", "n": len(A), "ciTBI": int(A["ciTBI"].sum()), "ciTBI_pct": 100 * A["ciTBI"].mean(),
        "ci_lo": cp_ci(int(A["ciTBI"].sum()), len(A))[0], "ci_hi": cp_ci(int(A["ciTBI"].sum()), len(A))[1], "CT_rate_pct": 100 * (A["CTDone"] == 1).mean(),
        "imaged_n": len(B), "TBIonCT": int((B["PosCT"] == 1).sum()), "TBIonCT_pct": 100 * (B["PosCT"] == 1).mean()}]
t2 += risk_rows("age_grp", ["0–2 months", "3–5 months", "6–11 months", "12–23 months"])
t2 += risk_rows("hema_cat", ["No hematoma", "Frontal hematoma", "Non-frontal hematoma, ≤3 cm", "Non-frontal hematoma, >3 cm", "Hematoma, location/size unknown"])
A["mech_lab"] = np.where(A["severe_mech"] == 1, "Severe", "Not severe"); t2 += risk_rows("mech_lab", ["Severe", "Not severe"])
A["loc_lab"] = np.where(A["loc5"] == 1, "Yes", "No"); t2 += risk_rows("loc_lab", ["Yes", "No"])
A["act_lab"] = np.where(A["not_normal"] == 1, "Not acting normally", "Acting normally"); t2 += risk_rows("act_lab", ["Not acting normally", "Acting normally"])
t2 += risk_rows("n_find_cat", ["1 finding", "≥2 findings"])
# combined strata (descriptive)
A["stratum"] = np.select([(A["hema_cat"] == "Non-frontal hematoma, >3 cm") | (A["AgeInMonth"] < 6) | (A["n_find"] >= 2),
                          (A["AgeInMonth"] >= 12) & (A["hema_cat"] != "Non-frontal hematoma, >3 cm") & (A["n_find"] == 1)],
                         ["Higher: age <6 months, or non-frontal hematoma >3 cm, or ≥2 findings", "Lower: age ≥12 months, single finding, no non-frontal hematoma >3 cm"],
                         "Intermediate: remaining infants")
t2 += risk_rows("stratum", ["Higher: age <6 months, or non-frontal hematoma >3 cm, or ≥2 findings", "Intermediate: remaining infants", "Lower: age ≥12 months, single finding, no non-frontal hematoma >3 cm"])
t2 += risk_rows("score", list(range(0, int(A["score"].max()) + 1)))
pd.DataFrame(t2).to_csv(os.path.join(OUT, "table2_risk.csv"), index=False)
for r in t2: P(f"{r['group']:>10} {r['level']:<70} ciTBI {r['ciTBI']}/{r['n']} ({r['ciTBI_pct']:.2f}%; {r['ci_lo']:.2f}–{r['ci_hi']:.2f})  CT {r['CT_rate_pct']:.1f}%  TBIonCT {r['TBIonCT']}/{r['imaged_n']} ({r['TBIonCT_pct']:.1f}%)")
# age x hematoma cross table
ct_rows = []
for ag in ["0–2 months", "3–5 months", "6–11 months", "12–23 months"]:
    for hc in ["No hematoma", "Frontal hematoma", "Non-frontal hematoma, ≤3 cm", "Non-frontal hematoma, >3 cm"]:
        a = A[(A["age_grp"] == ag) & (A["hema_cat"] == hc)]; b = a[a["CTDone"] == 1]
        ct_rows.append({"age_grp": ag, "hema_cat": hc, "n": len(a), "ciTBI": int(a["ciTBI"].sum()), "ciTBI_pct": 100 * a["ciTBI"].mean() if len(a) else np.nan,
                        "ci_lo": cp_ci(int(a["ciTBI"].sum()), len(a))[0], "ci_hi": cp_ci(int(a["ciTBI"].sum()), len(a))[1], "CT_rate_pct": 100 * (a["CTDone"] == 1).mean() if len(a) else np.nan,
                        "imaged_n": len(b), "TBIonCT": int((b["PosCT"] == 1).sum()), "TBIonCT_pct": 100 * (b["PosCT"] == 1).mean() if len(b) else np.nan})
ctab = pd.DataFrame(ct_rows); ctab.to_csv(os.path.join(OUT, "table_age_x_hematoma.csv"), index=False)

# exploratory multivariable logistic regression for ciTBI (cohort A), bootstrap CIs
Xa = pd.DataFrame({"age_lt6": (A["AgeInMonth"] < 6).astype(int), "nonfrontal_large": (A["hema_cat"] == "Non-frontal hematoma, >3 cm").astype(int),
                   "two_or_more": (A["n_find"] >= 2).astype(int), "severe_mech": A["severe_mech"]}); ya = A["ciTBI"].values
def fit_or(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y); return np.exp(m.coef_[0])
ors = fit_or(Xa, ya); rng = np.random.RandomState(SEED); boots = []
for _ in range(1000):
    i = rng.randint(0, len(ya), len(ya))
    if ya[i].sum() < 5: continue
    try: boots.append(fit_or(Xa.iloc[i], ya[i]))
    except Exception: pass
boots = np.array(boots); or_tab = pd.DataFrame({"predictor": Xa.columns, "OR": ors, "ci_lo": np.percentile(boots, 2.5, axis=0), "ci_hi": np.percentile(boots, 97.5, axis=0)})
or_tab.to_csv(os.path.join(OUT, "ciTBI_multivariable_OR.csv"), index=False); P("ciTBI ORs:\n" + or_tab.round(2).to_string(index=False))
auc_a = roc_auc_score(ya, LogisticRegression(C=1e6, max_iter=5000).fit(Xa, ya).predict_proba(Xa)[:, 1]); P("ciTBI 4-predictor model apparent AUC", round(auc_a, 3))
auc_score_A = roc_auc_score(ya, A["score"]); P("point score AUC for ciTBI in cohort A", round(auc_score_A, 3))

# ----------------------------------------------------------------------------- Cohort B: models for TBI on CT
yb = B["PosCT"].astype(int).values
def feats(d):
    X = pd.DataFrame(index=d.index)
    X["AgeInMonth"] = d["AgeInMonth"]; X["HemaLoc_2"] = (d["HemaLoc"] == 2).astype(float).where(d["HemaLoc"].notna())
    X["HemaLoc_3"] = (d["HemaLoc"] == 3).astype(float).where(d["HemaLoc"].notna()); X["HemaSize"] = d["HemaSize"].replace(92, 0); X["High_impact_InjSev"] = d["High_impact_InjSev"]
    F5 = list(X.columns)
    X["LocLen"] = d["LocLen"].replace(92, 0); X["ActNorm"] = d["ActNorm"]; X["HASeverity"] = d["HASeverity"].replace(92, 0); X["VomitNbr"] = d["VomitNbr"].replace(92, 0)
    X["FontBulg"] = d["FontBulg"]; X["Clav"] = d["Clav"]; X["NeuroD"] = d["NeuroD"]; X["OSI"] = d["OSI"]; X["Gender"] = d["Gender"].replace({1: 0, 2: 1})
    X["HemaLoc_92"] = (d["HemaLoc"] == 92).astype(float).where(d["HemaLoc"].notna())
    for v in [1, 2, 3]: X[f"SeizOccur_{v}"] = (d["SeizOccur"] == v).astype(float).where(d["SeizOccur"].notna())
    F14 = list(X.columns)
    for k, name in {1: "mvc_occ", 2: "pedestrian", 3: "bike_mv", 4: "bike_fall", 5: "wheeled", 6: "fall_ground", 7: "walk_object", 8: "fall_elev", 9: "fall_stairs", 10: "sports", 11: "assault", 12: "object_struck", 90: "other"}.items():
        X[f"mech_{name}"] = (d["InjuryMech"] == k).astype(float).where(d["InjuryMech"].notna())
    X["LOC_yes"] = (d["LOCSeparate"] == 1).astype(float).where(d["LOCSeparate"].notna()); X["LOC_susp"] = (d["LOCSeparate"] == 2).astype(float).where(d["LOCSeparate"].notna())
    X["SeizLen"] = d["SeizLen"].replace(92, 0); X["VomitStart"] = d["VomitStart"].replace(92, 0); X["VomitLast"] = d["VomitLast"].replace(92, 0); X["SFxBas"] = d["SFxBas"]
    for c in ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar", "ClavTem"]: X[c] = d[c].replace(92, 0)
    X["Observed"] = d["Observed"]; X["n_find"] = d["n_find"]; X["age_lt3m"] = (d["AgeInMonth"] < 3).astype(int)
    X["hema_pt_large"] = ((d["HemaLoc"] == 3) & (d["HemaSize"] == 3)).astype(int); X["hema_pt_medium"] = ((d["HemaLoc"] == 3) & (d["HemaSize"] == 2)).astype(int)
    X["hema_occ_large"] = ((d["HemaLoc"] == 2) & (d["HemaSize"] == 3)).astype(int); X["hema_nonfrontal_size"] = np.where(d["HemaLoc"].isin([2, 3]), d["HemaSize"].replace(92, 0), 0)
    X["age_x_pt"] = d["AgeInMonth"] * (d["HemaLoc"] == 3).astype(int)
    F51 = list(X.columns)
    return X, {"F5": F5, "F14": F14, "F51": F51}
Xb, FS = feats(B)
def LR(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
models = {"LR5": ("F5", LR), "LR14": ("F14", LR), "LR51": ("F51", LR),
          "RF14": ("F14", lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=-1))),
          "LGBM14": ("F14", lambda: make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, verbose=-1)))}
def spec_at(y, p, target):
    pos = np.sort(p[y == 1]); thr = pos[int(np.floor((1 - target) * len(pos)))]; pred = p >= thr
    return float((~pred[y == 0]).mean()), float((~pred).mean())
cmp_rows = []; oof_store = {}
for name, (fs, mk) in models.items():
    cols = FS[fs]; aucs = []; oofs = []
    for r in range(args.repeats):
        oof = np.zeros(len(yb))
        for a, v in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xb, yb):
            m = mk().fit(Xb.iloc[a][cols], yb[a]); oof[v] = m.predict_proba(Xb.iloc[v][cols])[:, 1]
        aucs.append(roc_auc_score(yb, oof)); oofs.append(oof)
    p = np.mean(oofs, axis=0); oof_store[name] = p
    rng = np.random.RandomState(SEED); b = [roc_auc_score(yb[i], p[i]) for i in (rng.randint(0, len(yb), len(yb)) for _ in range(2000))]
    s95 = spec_at(yb, p, 0.95); s90 = spec_at(yb, p, 0.90)
    m_all = mk().fit(Xb[cols], yb); app_auc = roc_auc_score(yb, m_all.predict_proba(Xb[cols])[:, 1])
    cmp_rows.append({"model": name, "n_features": len(cols), "apparent_auc": app_auc, "cv_auc_mean": np.mean(aucs), "cv_auc_sd": np.std(aucs), "boot_lo": np.percentile(b, 2.5), "boot_hi": np.percentile(b, 97.5),
                     "spec95": s95[0], "negfrac95": s95[1], "spec90": s90[0], "negfrac90": s90[1], "brier": brier_score_loss(yb, p)})
    P(f"{name:<7} feats={len(cols):>2} CV AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} (boot {np.percentile(b,2.5):.3f}–{np.percentile(b,97.5):.3f}) spec95 {s95[0]:.3f} neg95 {s95[1]:.3f} spec90 {s90[0]:.3f} neg90 {s90[1]:.3f}")
# point score in cohort B
sc = B["score"].values.astype(float); rng = np.random.RandomState(SEED)
b = [roc_auc_score(yb[i], oof_sc[i]) for i in (rng.randint(0, len(yb), len(yb)) for _ in range(2000))]
s95 = spec_at(yb, oof_sc, 0.95); s90 = spec_at(yb, oof_sc, 0.90)
cmp_rows.append({"model": "PointScore", "n_features": 4, "apparent_auc": roc_auc_score(yb, sc), "cv_auc_mean": roc_auc_score(yb, oof_sc), "cv_auc_sd": np.nan, "boot_lo": np.percentile(b, 2.5), "boot_hi": np.percentile(b, 97.5),
                 "spec95": s95[0], "negfrac95": s95[1], "spec90": s90[0], "negfrac90": s90[1], "brier": np.nan})
P(f"PointScore AUC {roc_auc_score(yb, sc):.3f} (boot {np.percentile(b,2.5):.3f}–{np.percentile(b,97.5):.3f}) spec95 {s95[0]:.3f} neg95 {s95[1]:.3f} spec90 {s90[0]:.3f} neg90 {s90[1]:.3f}")
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(os.path.join(OUT, "table3_model_comparison.csv"), index=False)

# LR5: coefficients on full cohort B, calibration of OOF, decision curve, per-100 outcomes
p5 = oof_store["LR5"]
m5 = LR().fit(Xb[FS["F5"]], yb); scl = m5.named_steps["standardscaler"]; lr5 = m5.named_steps["logisticregression"]
coef = lr5.coef_[0] / scl.scale_; b0 = lr5.intercept_[0] - np.sum(coef * scl.mean_)
coef_tab = pd.DataFrame({"variable": FS["F5"], "coef": coef, "OR": np.exp(coef)})
# bootstrap CI for ORs
rng = np.random.RandomState(SEED); bo = []
for _ in range(1000):
    i = rng.randint(0, len(yb), len(yb)); mm = LR().fit(Xb[FS["F5"]].iloc[i], yb[i]); s_ = mm.named_steps["standardscaler"]; bo.append(mm.named_steps["logisticregression"].coef_[0] / s_.scale_)
bo = np.exp(np.array(bo)); coef_tab["OR_lo"] = np.percentile(bo, 2.5, axis=0); coef_tab["OR_hi"] = np.percentile(bo, 97.5, axis=0)
coef_tab.to_csv(os.path.join(OUT, "lr5_coefficients.csv"), index=False); P("LR5 intercept", round(float(b0), 3), "\n" + coef_tab.round(3).to_string(index=False))
# calibration: slope/intercept via logistic recalibration of OOF logits; decile plot
logit = np.log(p5 / (1 - p5)); cal = LogisticRegression(C=1e6, max_iter=2000).fit(logit.reshape(-1, 1), yb)
cal_slope = float(cal.coef_[0][0])
dec = pd.qcut(p5, 10, labels=False, duplicates="drop"); calib = pd.DataFrame({"dec": dec, "p": p5, "y": yb}).groupby("dec").agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
calib.to_csv(os.path.join(OUT, "lr5_calibration_deciles.csv"), index=False)
# decision curve & per-100 (LR5 and score levels)
prev = yb.mean(); pts = np.round(np.arange(0.01, 0.16, 0.005), 3)
def nb(y, p, pt): pred = p >= pt; n = len(y); return (pred & (y == 1)).sum() / n - (pred & (y == 0)).sum() / n * pt / (1 - pt)
dca = pd.DataFrame({"pt": pts, "NB_all": [prev - (1 - prev) * pt / (1 - pt) for pt in pts], "NB_LR5": [nb(yb, p5, pt) for pt in pts], "NB_LR14": [nb(yb, oof_store["LR14"], pt) for pt in pts],
                    "LR5_CT_avoided_per100": [100 * (p5 < pt).mean() for pt in pts], "LR5_TBI_missed_per100": [100 * ((p5 < pt) & (yb == 1)).sum() / len(yb) for pt in pts],
                    "LR5_sens": [(p5 >= pt)[yb == 1].mean() for pt in pts], "LR5_spec": [(p5 < pt)[yb == 0].mean() for pt in pts], "LR5_NPV": [(yb[p5 < pt] == 0).mean() if (p5 < pt).sum() else np.nan for pt in pts]})
dca["ciTBI_below_threshold"] = [int(B["ciTBI"].values[p5 < pt].sum()) for pt in pts]; dca["n_below_threshold"] = [int((p5 < pt).sum()) for pt in pts]
dca["ciTBI_below_pct"] = [100 * B["ciTBI"].values[p5 < pt].mean() if (p5 < pt).sum() else np.nan for pt in pts]
dca["ciTBI_below_lo"] = [cp_ci(int(B["ciTBI"].values[p5 < pt].sum()), int((p5 < pt).sum()))[0] if (p5 < pt).sum() else np.nan for pt in pts]
dca["ciTBI_below_hi"] = [cp_ci(int(B["ciTBI"].values[p5 < pt].sum()), int((p5 < pt).sum()))[1] if (p5 < pt).sum() else np.nan for pt in pts]
dca["intervention_below_threshold"] = [int(((B["Neurosurgery"] == 1) | (B["Intub24Head"] == 1) | (B["DeathTBI"] == 1)).values[p5 < pt].sum()) for pt in pts]
dca.to_csv(os.path.join(OUT, "table4_decision_curve.csv"), index=False)
P("per-100 table:\n" + dca[dca["pt"].isin([0.02, 0.03, 0.04, 0.05, 0.06, 0.08])].round(3).to_string(index=False))
# score levels: TBI on CT (B) and ciTBI (A), cumulative ("score <= k")
sc_rows = []
for k in range(0, int(A["score"].max()) + 1):
    bB = B[B["score"] == k]; aA = A[A["score"] == k]; cB = B[B["score"] <= k]; cA = A[A["score"] <= k]
    sc_rows.append({"score": k, "B_n": len(bB), "B_TBIonCT": int((bB["PosCT"] == 1).sum()), "B_pct": 100 * (bB["PosCT"] == 1).mean() if len(bB) else np.nan,
                    "A_n": len(aA), "A_ciTBI": int(aA["ciTBI"].sum()), "A_pct": 100 * aA["ciTBI"].mean() if len(aA) else np.nan, "A_CT_rate": 100 * (aA["CTDone"] == 1).mean() if len(aA) else np.nan,
                    "cum_B_n": len(cB), "cum_B_TBIonCT": int((cB["PosCT"] == 1).sum()), "cum_B_pct": 100 * (cB["PosCT"] == 1).mean(), "cum_B_frac": 100 * len(cB) / len(B),
                    "cum_A_n": len(cA), "cum_A_ciTBI": int(cA["ciTBI"].sum()), "cum_A_pct": 100 * cA["ciTBI"].mean(), "cum_A_ci_lo": cp_ci(int(cA["ciTBI"].sum()), len(cA))[0], "cum_A_ci_hi": cp_ci(int(cA["ciTBI"].sum()), len(cA))[1], "cum_A_frac": 100 * len(cA) / len(A)})
sct = pd.DataFrame(sc_rows); sct.to_csv(os.path.join(OUT, "table5_score_levels.csv"), index=False); P("score levels:\n" + sct.round(2).to_string(index=False))
json.dump({"prevalence_B": float(prev), "lr5_intercept": float(b0), "cal_slope": cal_slope, "auc_score_A": float(auc_score_A), "auc_ciTBI_model_A": float(auc_a)}, open(os.path.join(OUT, "misc.json"), "w"), indent=2)


# ----------------------------------------------------------------------------- safety-oriented criteria (cohort A): rule-in and observation-eligible
sym = (A["Vomit"] == 1) | (A["Seiz"] == 1) | A["loc5"].astype(bool) | A["not_normal"].astype(bool)
nf_large = A["hema_cat"] == "Non-frontal hematoma, >3 cm"
rules = {
    "Rule-in A: higher-risk stratum (age <6 mo, or non-frontal hematoma >3 cm, or ≥2 findings)": A["stratum"].str.startswith("Higher"),
    "Rule-in B: higher-risk stratum, or seizure, or ≥3 vomiting episodes": A["stratum"].str.startswith("Higher") | (A["Seiz"] == 1) | (A["VomitNbr"] == 3),
    "Rule-in C: score ≥ 3": A["score"] >= 3,
    "Rule-in D: score ≥ 1": A["score"] >= 1,
    "Rule-in E: age <12 mo, or non-frontal hematoma, or vomiting, or seizure, or LOC ≥5 s": (A["AgeInMonth"] < 12) | A["nonfrontal"].astype(bool) | (A["Vomit"] == 1) | (A["Seiz"] == 1) | A["loc5"].astype(bool),
}
obs = {
    "Observation-eligible 1: age ≥12 mo, single finding, no non-frontal hematoma >3 cm (lower-risk stratum)": A["stratum"].str.startswith("Lower"),
    "Observation-eligible 2: asymptomatic (no vomiting, seizure, LOC ≥5 s, acting normally), age ≥12 mo, no non-frontal hematoma >3 cm": (~sym) & (A["AgeInMonth"] >= 12) & (~nf_large),
    "Observation-eligible 3: asymptomatic, age ≥6 mo, no non-frontal hematoma": (~sym) & (A["AgeInMonth"] >= 6) & (~A["nonfrontal"].astype(bool)),
    "Observation-eligible 4: asymptomatic and score ≤1": (~sym) & (A["score"] <= 1),
}
rows = []
for name, mask in list(rules.items()) + list(obs.items()):
    flagged = A[mask]; rest = A[~mask]; kf = int(flagged["ciTBI"].sum()); kr = int(rest["ciTBI"].sum())
    if name.startswith("Rule-in"):
        rows.append({"criterion": name, "type": "rule-in", "n_flagged": len(flagged), "pct_flagged": 100 * len(flagged) / len(A), "ciTBI_captured": kf, "ciTBI_total": int(A["ciTBI"].sum()),
                     "sens_pct": 100 * kf / A["ciTBI"].sum(), "sens_lo": cp_ci(kf, int(A["ciTBI"].sum()))[0], "sens_hi": cp_ci(kf, int(A["ciTBI"].sum()))[1],
                     "neurosurgery_captured": int((flagged["Neurosurgery"] == 1).sum()), "ciTBI_in_unflagged": kr, "risk_unflagged_pct": 100 * kr / len(rest) if len(rest) else np.nan,
                     "risk_unflagged_lo": cp_ci(kr, len(rest))[0], "risk_unflagged_hi": cp_ci(kr, len(rest))[1], "current_CT_rate_flagged_pct": 100 * (flagged["CTDone"] == 1).mean()})
    else:
        rows.append({"criterion": name, "type": "observation-eligible", "n_flagged": len(flagged), "pct_flagged": 100 * len(flagged) / len(A), "ciTBI_captured": kr, "ciTBI_total": int(A["ciTBI"].sum()),
                     "sens_pct": 100 * kr / A["ciTBI"].sum(), "sens_lo": cp_ci(kr, int(A["ciTBI"].sum()))[0], "sens_hi": cp_ci(kr, int(A["ciTBI"].sum()))[1],
                     "neurosurgery_captured": int((rest["Neurosurgery"] == 1).sum()), "ciTBI_in_unflagged": kf, "risk_unflagged_pct": 100 * kf / len(flagged) if len(flagged) else np.nan,
                     "risk_unflagged_lo": cp_ci(kf, len(flagged))[0], "risk_unflagged_hi": cp_ci(kf, len(flagged))[1], "current_CT_rate_flagged_pct": 100 * (flagged["CTDone"] == 1).mean()})
saf = pd.DataFrame(rows); saf.to_csv(os.path.join(OUT, "table_safety_criteria.csv"), index=False); P("safety criteria:\n" + saf.round(2).to_string(index=False))
# exhaustive search: minimum fraction flagged by any OR-combination of up to 4 bedside items that captures all (or all but one) ciTBI
import itertools
comps = {"age<3": A["AgeInMonth"] < 3, "age<6": A["AgeInMonth"] < 6, "age<12": A["AgeInMonth"] < 12, "nf>3cm": nf_large, "nf≥1cm": A["nonfrontal"].astype(bool) & (A["HemaSize"] >= 2), "nf_any": A["nonfrontal"].astype(bool),
         "any_hematoma": A["Hema"] == 1, "≥2findings": A["n_find"] >= 2, "vomiting": A["Vomit"] == 1, "vomit≥3": A["VomitNbr"] == 3, "seizure": A["Seiz"] == 1, "LOC≥5s": A["loc5"].astype(bool), "not_normal": A["not_normal"].astype(bool), "severe_mech": A["severe_mech"].astype(bool)}
best = {30: (101.0, None), 29: (101.0, None)}
for r in range(1, 5):
    for combo in itertools.combinations(comps, r):
        mk = np.zeros(len(A), bool)
        for k in combo: mk |= comps[k].values
        tp = int((mk & (A["ciTBI"].values == 1)).sum()); fl = 100 * mk.mean()
        for target in (30, 29):
            if tp >= target and fl < best[target][0]: best[target] = (fl, combo)
jc2 = json.load(open(os.path.join(OUT, "japan_context.json"))) if os.path.exists(os.path.join(OUT, "japan_context.json")) else {}
jc2["min_flagged_for_all_ciTBI_pct"] = best[30][0]; jc2["min_flagged_for_all_ciTBI_rule"] = " OR ".join(best[30][1]); jc2["min_flagged_for_29of30_pct"] = best[29][0]; jc2["min_flagged_for_29of30_rule"] = " OR ".join(best[29][1])
# ciTBI cases not captured by rule-in B and by observation-eligible 2
def case_desc(d): return [{"age_months": int(r["AgeInMonth"]), "hematoma": r["hema_cat"], "severe_mech": int(r["severe_mech"]), "loc5": int(r["loc5"]), "not_normal": int(r["not_normal"]), "vomiting": int(r["Vomit"] == 1), "seizure": int(r["Seiz"] == 1), "n_find": int(r["n_find"]), "score": int(r["score"]), "neurosurgery": int(r["Neurosurgery"] == 1)} for _, r in d.iterrows()]
jc2["ciTBI_not_captured_by_ruleB"] = case_desc(A[(~rules["Rule-in B: higher-risk stratum, or seizure, or ≥3 vomiting episodes"]) & (A["ciTBI"] == 1)])
jc2["ciTBI_in_obs2"] = case_desc(A[obs["Observation-eligible 2: asymptomatic (no vomiting, seizure, LOC ≥5 s, acting normally), age ≥12 mo, no non-frontal hematoma >3 cm"] & (A["ciTBI"] == 1)])
jc2["symptomatic_n"] = int(sym.sum()); jc2["symptomatic_ciTBI"] = int(A[sym]["ciTBI"].sum())
json.dump(jc2, open(os.path.join(OUT, "japan_context.json"), "w"), indent=2, ensure_ascii=False)
P("min flagged for 30/30: %.1f%% (%s); for 29/30: %.1f%% (%s)" % (best[30][0], best[30][1], best[29][0], best[29][1]))

# ----------------------------------------------------------------------------- figures
# Fig 1 flow
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.5, 8.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
def box(x, y, w, h, t, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec="black", lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.4)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
mx, mw = 0.6, 5.0
box(mx, 10.6, mw, 1.1, f"PECARN TBI public use dataset\nn = {flow['total']:,}", 10.5, True)
box(mx, 8.9, mw, 1.1, f"Age < 2 years\nn = {flow['age_lt2']:,}", 10.5, True)
box(mx, 7.0, mw, 1.3, f"GCS 15, no altered mental status,\nno palpable skull fracture\nn = {flow['gcs15_noAMS_noSFx']:,}", 10.5, True)
box(mx, 4.7, mw, 1.7, f"Cohort A: intermediate-risk infants\n(≥1 of: severe mechanism, LOC ≥5 s,\nnon-frontal hematoma, not acting normally)\nn = {flow['cohortA']:,}; ciTBI {flow['ciTBI']} ({100*flow['ciTBI']/flow['cohortA']:.2f}%)", 10, True)
box(mx, 2.4, mw, 1.6, f"Cohort B: head CT performed in the ED\nn = {flow['cohortB']:,} ({100*flow['cohortB']/flow['cohortA']:.1f}%)\nTBI on CT {flow['TBIonCT']} ({100*flow['TBIonCT']/flow['cohortB']:.1f}%); ciTBI {flow['ciTBI_in_B']}", 10, True)
for y1, y2 in [(10.6, 10.0), (8.9, 8.3), (7.0, 6.4), (4.7, 4.0)]: arrow(mx + mw / 2, y1, mx + mw / 2, y2)
sx, sw = 6.4, 3.3
def side(y, t): box(sx, y, sw, 1.0, t, 9); arrow(mx + mw, y + 0.5, sx, y + 0.5)
side(9.0, f"Excluded: age ≥ 2 years\nn = {flow['total']-flow['age_lt2']:,}")
side(7.15, f"Excluded: GCS < 15, AMS,\npalpable skull fracture or missing\n(high risk) n = {flow['excluded_high_or_missing']:,}")
side(5.05, f"Excluded: no intermediate-risk\nfinding (very low risk)\nn = {flow['excluded_very_low']:,}")
side(2.7, f"Not imaged (observed/discharged)\nn = {flow['not_imaged']:,}; ciTBI {flow['ciTBI_not_imaged']}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure1_flow.png"), dpi=300, bbox_inches="tight"); plt.close()
# Fig 2 ciTBI risk by age x hematoma (grouped bars with exact CI)
fig, ax = plt.subplots(figsize=(9, 5)); ages = ["0–2 months", "3–5 months", "6–11 months", "12–23 months"]; hcs = ["No hematoma", "Frontal hematoma", "Non-frontal hematoma, ≤3 cm", "Non-frontal hematoma, >3 cm"]
w = 0.2; hatches_ = ["", "////", "....", "xxxx"]  # white bars with hatching (journal style: no tints of shading)
for j, hc in enumerate(hcs):
    sub = ctab[ctab["hema_cat"] == hc].set_index("age_grp").loc[ages]
    x = np.arange(len(ages)) + (j - 1.5) * w
    ax.bar(x, sub["ciTBI_pct"], w, color="white", edgecolor="black", linewidth=0.7, hatch=hatches_[j], label=hc, yerr=[sub["ciTBI_pct"] - sub["ci_lo"], sub["ci_hi"] - sub["ciTBI_pct"]], capsize=2, error_kw=dict(lw=0.8))
    for xi, (n_, k_) in zip(x, zip(sub["n"], sub["ciTBI"])): ax.text(xi, -1.6, f"{k_}/{n_}", ha="center", va="top", fontsize=6.5, rotation=90)
ax.set_xticks(np.arange(len(ages))); ax.set_xticklabels(ages); ax.set_ylabel("Clinically important TBI, % (95% CI)"); ax.set_ylim(-9, max(ctab["ci_hi"].max() + 3, 20))
ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=8, loc="upper right"); ax.set_title("Cohort A: ciTBI by age and scalp hematoma (n = %d)" % len(A), fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure2_ciTBI_age_hematoma.png"), dpi=300, bbox_inches="tight"); plt.close()
# Fig 3 (a) ROC, (b) calibration, (c) decision curve for LR5
plt.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(11, 9.6)); gs = fig.add_gridspec(2, 2); axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
fpr, tpr, _ = roc_curve(yb, p5); r5 = cmp[cmp.model == "LR5"].iloc[0]
axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"4-variable model, AUC {r5.cv_auc_mean:.2f} ({r5.boot_lo:.2f}–{r5.boot_hi:.2f})")
fpr2, tpr2, _ = roc_curve(yb, oof_sc); rs = cmp[cmp.model == "PointScore"].iloc[0]; axes[0].plot(fpr2, tpr2, color="steelblue", lw=1.5, ls="--", label=f"Point score, AUC {rs.cv_auc_mean:.2f} ({rs.boot_lo:.2f}–{rs.boot_hi:.2f})")
axes[0].plot([0, 1], [0, 1], "k:", lw=1); axes[0].set_xlabel("1 − specificity"); axes[0].set_ylabel("Sensitivity"); axes[0].legend(fontsize=10, loc="lower right"); axes[0].set_title("Out-of-fold ROC (10 × 5-fold CV)", fontsize=12)
axes[1].plot([0, 0.3], [0, 0.3], "k:", lw=1); axes[1].errorbar(calib["mean_pred"], calib["obs"], yerr=[calib["obs"] - np.array([cp_ci(int(round(o * n)), n)[0] / 100 for o, n in zip(calib["obs"], calib["n"])]), np.array([cp_ci(int(round(o * n)), n)[1] / 100 for o, n in zip(calib["obs"], calib["n"])]) - calib["obs"]], fmt="o", color="darkorange", capsize=2)
axes[1].set_xlabel("Predicted probability of TBI on CT (decile mean)"); axes[1].set_ylabel("Observed proportion"); axes[1].set_title(f"Calibration (slope {cal_slope:.2f})", fontsize=12); axes[1].set_xlim(0, 0.3); axes[1].set_ylim(0, 0.35)
ax = axes[2]; ax.plot(pts, dca["NB_all"], "k-", lw=1.5, label="CT for all intermediate-risk infants"); ax.plot(pts, [0] * len(pts), "k--", lw=1, label="CT for none")
ax.plot(pts, dca["NB_LR5"], color="darkorange", lw=2, label="4-variable logistic model"); ax.plot(pts, dca["NB_LR14"], color="steelblue", lw=1.5, ls="--", label="14-variable logistic model")
ax.set_xlabel("Threshold probability of TBI on CT"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.005, prev + 0.005); ax.legend(fontsize=10); ax.set_title("Decision curve (out-of-fold probabilities)", fontsize=12)
for a_, l, x_ in zip(axes, ["a", "b", "c"], [-0.08, -0.08, -0.035]): a_.text(x_, 1.04, l, transform=a_.transAxes, fontsize=16, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure3_performance.png"), dpi=300, bbox_inches="tight"); plt.close()
plt.rcParams.update({"font.size": 10})

# ----------------------------------------------------------------------------- context for the Japan-focused discussion
jc = json.load(open(os.path.join(OUT, "japan_context.json"))) if os.path.exists(os.path.join(OUT, "japan_context.json")) else {}
sf = []
for name, mask in [("Severe mechanism only", (A["n_find"] == 1) & (A["severe_mech"] == 1)), ("Non-frontal hematoma only", (A["n_find"] == 1) & (A["nonfrontal"] == 1)),
                   ("Loss of consciousness ≥5 s only", (A["n_find"] == 1) & (A["loc5"] == 1)), ("Not acting normally only", (A["n_find"] == 1) & (A["not_normal"] == 1))]:
    a = A[mask]; k = int(a["ciTBI"].sum())
    sf.append({"finding": name, "n": len(a), "CT_rate_pct": 100 * (a["CTDone"] == 1).mean(), "ciTBI": k, "ciTBI_pct": 100 * k / len(a), "ci_lo": cp_ci(k, len(a))[0], "ci_hi": cp_ci(k, len(a))[1]})
pd.DataFrame(sf).to_csv(os.path.join(OUT, "single_finding_ct_and_ciTBI.csv"), index=False); jc["single_findings"] = sf
low = A[A["stratum"].str.startswith("Lower")]
cases = []
for _, r in low[low["ciTBI"] == 1].iterrows():
    cases.append({"age_months": int(r["AgeInMonth"]), "finding": "severe mechanism" if r["severe_mech"] else ("LOC ≥5 s" if r["loc5"] else ("non-frontal hematoma" if r["nonfrontal"] else "not acting normally")),
                  "hematoma": r["hema_cat"], "vomiting": int(r["Vomit"] == 1), "seizure": int(r["Seiz"] == 1), "neurosurgery": int(r["Neurosurgery"] == 1), "intubation": int(r["Intub24Head"] == 1), "death": int(r["DeathTBI"] == 1), "admission_2nights": int(r["HospHeadPosCT"] == 1), "score": int(r["score"])})
jc["lower_stratum_ciTBI_cases"] = cases
jc["lower_stratum"] = {"n": len(low), "CT": int((low["CTDone"] == 1).sum()), "observed": int((low["Observed"] == 1).sum())}
ns = A[A["Neurosurgery"] == 1]
jc["neurosurgery_cases"] = [{"age_months": int(r["AgeInMonth"]), "n_find": int(r["n_find"]), "stratum": r["stratum"].split(":")[0], "score": int(r["score"]), "vomiting_episodes": (int(r["VomitNbr"]) if r["VomitNbr"] in (1, 2, 3) else 0)} for _, r in ns.iterrows()]
jc["ciTBI_per_TBIonCT_ratio"] = float(A["ciTBI"].mean() / B["PosCT"].mean())
jc["threshold3_ciTBI_equiv_pct"] = float(3 * jc["ciTBI_per_TBIonCT_ratio"])
jc["score_le1"] = {"n": int((A["score"] <= 1).sum()), "CT": int(((A["score"] <= 1) & (A["CTDone"] == 1)).sum()), "ciTBI": int(A[A["score"] <= 1]["ciTBI"].sum())}
jc["ciTBI_score_distribution"] = {int(k): int(v) for k, v in A[A["ciTBI"] == 1]["score"].value_counts().sort_index().items()}
sed_known = B[B["CTSed"].isin([0, 1])]; jc["sedation"] = {"n_known": int(len(sed_known)), "sedated": int((sed_known["CTSed"] == 1).sum()), "pct": float(100 * (sed_known["CTSed"] == 1).mean()),
    "reason_agitation": int((B["CTSedAgitate"] == 1).sum()), "reason_age": int((B["CTSedAge"] == 1).sum()), "reason_request": int((B["CTSedRqst"] == 1).sum())}
low_img = low[low["CTDone"] == 1]; jc["lower_stratum_ct_per_ciTBI"] = float(len(low_img) / max(1, int(low["ciTBI"].sum())))
json.dump(jc, open(os.path.join(OUT, "japan_context.json"), "w"), indent=2, ensure_ascii=False); P("japan context:", json.dumps(jc, ensure_ascii=False))


# ----------------------------------------------------------------------------- proposed management pathway (hierarchical) and Figure 4
ruleB = rules["Rule-in B: higher-risk stratum, or seizure, or ≥3 vomiting episodes"]
obs2 = obs["Observation-eligible 2: asymptomatic (no vomiting, seizure, LOC ≥5 s, acting normally), age ≥12 mo, no non-frontal hematoma >3 cm"]
path = np.where(ruleB, "CT recommended", np.where(obs2, "Observation-eligible", "Individualized"))
path2 = np.where(ruleB, "CT recommended", "Observation")
A["pathway"] = path
sed_rate = jc.get("sedation", {}).get("pct", np.nan) / 100
pw = {}
for g in ["CT recommended", "Observation-eligible", "Individualized"]:
    a = A[A["pathway"] == g]; b = a[a["CTDone"] == 1]; k = int(a["ciTBI"].sum())
    pw[g] = {"n": int(len(a)), "pct": 100 * len(a) / len(A), "ciTBI": k, "ciTBI_pct": 100 * k / len(a), "ci_lo": cp_ci(k, len(a))[0], "ci_hi": cp_ci(k, len(a))[1],
             "neurosurgery": int((a["Neurosurgery"] == 1).sum()), "current_CT_pct": 100 * (a["CTDone"] == 1).mean(), "imaged_n": int(len(b)), "TBIonCT": int((b["PosCT"] == 1).sum()),
             "TBIonCT_pct_imaged": 100 * (b["PosCT"] == 1).mean() if len(b) else np.nan, "ciTBI_intervention": int(((a["Neurosurgery"] == 1) | (a["Intub24Head"] == 1) | (a["DeathTBI"] == 1)).sum())}
# per 100 observation-eligible infants managed by observation instead of current practice
o = pw["Observation-eligible"]
pw["per100_observation"] = {"CT_avoided": o["current_CT_pct"], "sedations_avoided": o["current_CT_pct"] * sed_rate, "ciTBI_expected": o["ciTBI_pct"], "ciTBI_intervention_expected": 100 * o["ciTBI_intervention"] / o["n"],
                            "TBIonCT_found_if_all_imaged_est": o["TBIonCT_pct_imaged"]}
# merged observation group (two-step pathway)
a = A[path2 == "Observation"]; b_ = a[a["CTDone"] == 1]; k = int(a["ciTBI"].sum())
pw["Observation_merged"] = {"n": int(len(a)), "pct": 100 * len(a) / len(A), "ciTBI": k, "ciTBI_pct": 100 * k / len(a), "ci_lo": cp_ci(k, len(a))[0], "ci_hi": cp_ci(k, len(a))[1],
    "neurosurgery": int((a["Neurosurgery"] == 1).sum()), "ciTBI_intervention": int(((a["Neurosurgery"] == 1) | (a["Intub24Head"] == 1) | (a["DeathTBI"] == 1)).sum()), "current_CT_pct": 100 * (a["CTDone"] == 1).mean(),
    "imaged_n": int(len(b_)), "TBIonCT": int((b_["PosCT"] == 1).sum()), "TBIonCT_pct_imaged": 100 * (b_["PosCT"] == 1).mean() if len(b_) else np.nan,
    "n_6_11mo": int((a["AgeInMonth"] < 12).sum()), "n_symptomatic": int(sym[path2 == "Observation"].sum()),
    "ciTBI_6_11mo": int(a[a["AgeInMonth"] < 12]["ciTBI"].sum()), "ciTBI_symptomatic": int(a[sym[path2 == "Observation"]]["ciTBI"].sum())}
pw["per100_observation_merged"] = {"CT_avoided": pw["Observation_merged"]["current_CT_pct"], "sedations_avoided": pw["Observation_merged"]["current_CT_pct"] * sed_rate, "ciTBI_expected": pw["Observation_merged"]["ciTBI_pct"]}
pw["overall_CT_rate_current_pct"] = 100 * (A["CTDone"] == 1).mean()
pw["pathway_CT_rate_if_CTgroup_all_imaged_and_obs_none_pct"] = pw["CT recommended"]["pct"] + pw["Individualized"]["pct"] * pw["Individualized"]["current_CT_pct"] / 100
json.dump(pw, open(os.path.join(OUT, "pathway.json"), "w"), indent=2); P("pathway:", json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in pw.items()}))
# Figure 4: two-step pathway diagram
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")
def fbox(x, y, w, h, t, fs=9.5, bold=False, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="black", lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)
def farrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
c = pw["CT recommended"]; o = pw["Observation_merged"]
fbox(3.5, 6.1, 5.0, 1.0, f"PECARN intermediate-risk infant < 2 years\n(Cohort A, n = {len(A):,}; ciTBI {int(A['ciTBI'].sum())}, {100*A['ciTBI'].mean():.2f}%)", 10, True)
fbox(3.5, 4.2, 5.0, 1.3, "Step 1  Any of: age <6 months, non-frontal hematoma >3 cm,\n≥2 intermediate-risk findings, post-traumatic seizure,\nor ≥3 vomiting episodes?", 9.5)
farrow(6.0, 6.1, 6.0, 5.5)
fbox(8.9, 3.95, 3.0, 1.8, f"CT recommended\n{c['n']:,} infants ({c['pct']:.0f}%)\nciTBI {c['ciTBI']}/{c['n']:,} ({c['ciTBI_pct']:.2f}%;\n95% CI {c['ci_lo']:.2f}–{c['ci_hi']:.2f}%)\nincl. all {c['neurosurgery']} neurosurgical cases\ncurrent CT rate {c['current_CT_pct']:.0f}%", 9, False, "#f4d7d7")
farrow(8.5, 4.85, 8.9, 4.85); ax.text(8.55, 5.0, "Yes", fontsize=9)
fbox(3.5, 1.3, 5.0, 2.2, f"Step 2  Observation with re-examination\n{o['n']:,} infants ({o['pct']:.0f}%)\nciTBI {o['ciTBI']}/{o['n']:,} ({o['ciTBI_pct']:.2f}%; 95% CI {o['ci_lo']:.2f}–{o['ci_hi']:.2f}%)\nciTBI cases: admitted without intervention\n(under actual care; all diagnosed by CT in the ED)\ncurrent CT rate {o['current_CT_pct']:.0f}%\nCT if symptoms develop or persist during observation", 9, False, "#d7ead7")
farrow(6.0, 4.2, 6.0, 3.5); ax.text(6.15, 3.8, "No", fontsize=9)
fbox(0.2, 1.65, 3.0, 1.7, f"Hypothetical estimate (not tested):\nif none of these infants were imaged,\nabout {o['current_CT_pct']:.0f} CT scans and\n{o['current_CT_pct']*sed_rate:.1f} sedations per 100 infants\nwould be avoided. Safety of deferring\nCT was not evaluated in this study.", 8.6, False, "#f2f2f2")
fbox(8.9, 1.5, 3.0, 2.0, f"Composition of the observation group\nage 6–11 months: {o['n_6_11mo']:,} (ciTBI {o['ciTBI_6_11mo']})\nage 12–23 months: {o['n']-o['n_6_11mo']:,} (ciTBI {o['ciTBI']-o['ciTBI_6_11mo']})\nwith vomiting, LOC or abnormal\nbehaviour: {o['n_symptomatic']:,} (ciTBI {o['ciTBI_symptomatic']})", 8.6, False, "white")
ax.text(0.2, 0.3, "Observation: about 6 h in the ED with re-examination and written return precautions (Japanese 2019 guidance); consider abusive head trauma in infants\n<6 months or with an unexplained mechanism. Risks are those observed under the care actually delivered. ciTBI, clinically important TBI; LOC, loss of consciousness.\nDerived from these data; not externally validated.", fontsize=8, va="bottom")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure4_pathway.png"), dpi=300, bbox_inches="tight"); plt.close()
json.dump(pw, open(os.path.join(OUT, "pathway.json"), "w"), indent=2)
P("DONE"); log.close()
