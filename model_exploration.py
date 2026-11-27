#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systematic exploration of better modelling strategies for TBI-on-CT prediction in PECARN
intermediate-risk infants (< 2 years) who underwent CT.

Compared dimensions
  1. Feature sets   F0 = the 14 original variables
                    F1 = F0 + additional pre-imaging variables (detailed mechanism, LOC history, seizure length,
                         vomiting timing, basilar skull fracture signs, sites of trauma above the clavicles,
                         observation before CT decision, number of intermediate-risk findings, age < 3 months,
                         hematoma location x size combinations)
                    F2 = F1 + clinician-recorded indications for CT (Ind* variables; exploratory only)
  2. Models         penalized logistic regression (L2 / elastic net), logistic regression with age splines,
                    random forest, extra trees, LightGBM, histogram gradient boosting, MLP, stacking (LR+LGBM),
                    simple point score (Dayan-like)
  3. Training pool  intermediate-risk infants only (n=1,349) vs. augmented with other imaged infants < 2 years
                    (very-low-risk imaged infants, and additionally high-risk imaged infants); evaluation is always
                    out-of-fold on the intermediate-risk cohort
Evaluation         repeated stratified 5-fold CV (R repeats) on the full intermediate cohort; pooled OOF predictions
                    per repeat -> AUC, specificity / fraction classified negative at 95% and 90% sensitivity,
                    Brier score, net benefit (decision curve) at threshold probabilities 2-10%.
                    In addition, a fixed 80/20 split (seed 42) is evaluated for the top models.
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="exploration")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); OUT = args.out; SEED = args.seed

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

log = open(os.path.join(OUT, "log.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# ----------------------------------------------------------------------------- cohorts
df = pd.read_csv(args.csv, low_memory=False)
u2 = df[(df["AgeTwoPlus"] == 1) & (df["CTDone"] == 1) & (df["GCSTotal"] >= 14)].copy()
# high risk (or unclassifiable): GCS < 15, AMS present or missing, palpable/unclear skull fracture or missing -> matches the main analysis (n = 1,349 intermediate)
u2["high_risk"] = ((u2["GCSTotal"] != 15) | (u2["AMS"] != 0) | (u2["SFxPalp"] != 0)).astype(int)
u2["n_inter"] = ((u2["High_impact_InjSev"] == 3).astype(int) + u2["LocLen"].isin([2, 3, 4]).astype(int)
                 + u2["HemaLoc"].isin([2, 3]).astype(int) + (u2["ActNorm"] == 0).astype(int))
u2["group"] = np.where(u2["high_risk"] == 1, "high", np.where(u2["n_inter"] >= 1, "intermediate", "verylow"))
inter = u2[u2["group"] == "intermediate"].copy()
P("cohort sizes:", u2["group"].value_counts().to_dict(), "| intermediate TBI:", int(inter["PosCT"].sum()))

# ----------------------------------------------------------------------------- features
def build_features(d):
    X = pd.DataFrame(index=d.index)
    def na92(c): return d[c].replace(92, 0)
    # ---- F0 (original 14)
    X["High_impact_InjSev"] = d["High_impact_InjSev"]
    X["LocLen"] = na92("LocLen"); X["ActNorm"] = d["ActNorm"]; X["HASeverity"] = na92("HASeverity")
    X["VomitNbr"] = na92("VomitNbr"); X["FontBulg"] = d["FontBulg"]; X["HemaSize"] = na92("HemaSize")
    X["Clav"] = d["Clav"]; X["NeuroD"] = d["NeuroD"]; X["OSI"] = d["OSI"]; X["AgeInMonth"] = d["AgeInMonth"]
    X["Gender"] = d["Gender"].replace({1: 0, 2: 1})
    for v in [2, 3, 92]: X[f"HemaLoc_{v}"] = (d["HemaLoc"] == v).astype(float).where(d["HemaLoc"].notna())
    for v in [1, 2, 3]: X[f"SeizOccur_{v}"] = (d["SeizOccur"] == v).astype(float).where(d["SeizOccur"].notna())
    F0 = list(X.columns)
    # ---- F1 extras (all recorded before imaging)
    mech_groups = {1: "mvc_occ", 2: "pedestrian", 3: "bike_mv", 4: "bike_fall", 5: "wheeled", 6: "fall_ground", 7: "walk_object",
                   8: "fall_elev", 9: "fall_stairs", 10: "sports", 11: "assault", 12: "object_struck", 90: "other"}
    for k, name in mech_groups.items():
        X[f"mech_{name}"] = (d["InjuryMech"] == k).astype(float).where(d["InjuryMech"].notna())
    X["LOC_yes"] = (d["LOCSeparate"] == 1).astype(float).where(d["LOCSeparate"].notna())
    X["LOC_susp"] = (d["LOCSeparate"] == 2).astype(float).where(d["LOCSeparate"].notna())
    X["SeizLen"] = na92("SeizLen"); X["VomitStart"] = na92("VomitStart"); X["VomitLast"] = na92("VomitLast")
    X["SFxBas"] = d["SFxBas"]
    for c in ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar", "ClavTem"]: X[c] = d[c].replace(92, 0)
    X["Observed"] = d["Observed"]
    X["n_inter"] = d["n_inter"]; X["age_lt3m"] = (d["AgeInMonth"] < 3).astype(int)
    X["hema_pt_large"] = ((d["HemaLoc"] == 3) & (d["HemaSize"] == 3)).astype(int)
    X["hema_pt_medium"] = ((d["HemaLoc"] == 3) & (d["HemaSize"] == 2)).astype(int)
    X["hema_occ_large"] = ((d["HemaLoc"] == 2) & (d["HemaSize"] == 3)).astype(int)
    X["hema_nonfrontal_size"] = np.where(d["HemaLoc"].isin([2, 3]), na92("HemaSize"), 0)
    X["age_x_pt"] = d["AgeInMonth"] * (d["HemaLoc"] == 3).astype(int)
    F1 = list(X.columns)
    # ---- F2: clinician indications for CT (exploratory)
    for c in ["IndAge", "IndAmnesia", "IndAMS", "IndClinSFx", "IndHA", "IndHema", "IndLOC", "IndMech", "IndNeuroD",
              "IndRqstMD", "IndRqstParent", "IndRqstTrauma", "IndSeiz", "IndVomit", "IndXraySFx", "IndOth"]:
        X[c] = d[c].replace(92, 0)
    F2 = list(X.columns)
    Fc = ["AgeInMonth", "HemaLoc_2", "HemaLoc_3", "HemaSize", "High_impact_InjSev"]
    return X, {"F0": F0, "F1": F1, "F2": F2, "Fc": Fc}

Xall, FS = build_features(u2)
yall = u2["PosCT"].astype(int).values
is_inter = (u2["group"] == "intermediate").values
is_verylow = (u2["group"] == "verylow").values
is_high = (u2["group"] == "high").values
P("n features:", {k: len(v) for k, v in FS.items()})

# ----------------------------------------------------------------------------- models
def pre(cols):
    return ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols)], remainder="drop")

def make_models(cols):
    age_idx = cols.index("AgeInMonth")
    other = [c for c in cols if c != "AgeInMonth"]
    spline_pre = ColumnTransformer([
        ("age", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=5, degree=3)), ["AgeInMonth"]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other)])
    m = {
        "LR_L2": make_pipeline(pre(cols), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")),
        "LR_L2_C1": make_pipeline(pre(cols), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
        "LR_elasticnet": make_pipeline(pre(cols), LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, max_iter=5000, class_weight="balanced")),
        "LR_age_spline": make_pipeline(spline_pre, LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")),
        "RandomForest": make_pipeline(pre(cols), RandomForestClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        "ExtraTrees": make_pipeline(pre(cols), ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        "LightGBM": make_pipeline(pre(cols), lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=3, random_state=SEED, verbose=-1)),
        "HistGB": make_pipeline(pre(cols), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0, class_weight="balanced", random_state=SEED)),
        "MLP": make_pipeline(pre(cols), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=600, early_stopping=True, random_state=SEED)),
    }
    m["Stack_LR+LGBM"] = StackingClassifier(
        estimators=[("lr", m["LR_L2"]), ("lgbm", m["LightGBM"])],
        final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="predict_proba", n_jobs=1)
    return m

def point_score(X):
    # Dayan-like simple score: age < 3 mo (2 pts), age 3-11 mo (1), non-frontal hematoma (1), large hematoma (1), medium (0.5), severe mechanism (1)
    return (2 * (X["AgeInMonth"] < 3) + 1 * ((X["AgeInMonth"] >= 3) & (X["AgeInMonth"] < 12))
            + 1 * (X["HemaLoc_2"].fillna(0) + X["HemaLoc_3"].fillna(0)) + 1 * (X["HemaSize"] == 3) + 0.5 * (X["HemaSize"] == 2)
            + 1 * (X["High_impact_InjSev"] == 3)).astype(float)

# ----------------------------------------------------------------------------- evaluation helpers
def spec_at_sens(y, p, target):
    pos = np.sort(p[y == 1]); k = int(np.floor((1 - target) * len(pos))); thr = pos[k]
    pred = p >= thr
    return float((~pred[y == 0]).mean()), float((~pred).mean()), float(pred[y == 1].mean())

def net_benefit(y, p, pt):
    pred = p >= pt; n = len(y)
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    return tp / n - fp / n * pt / (1 - pt)

def summarize(name, y, oof_list):
    rows = []
    for p in oof_list:
        pp = np.clip(p, 0, 1) if (p.min() >= 0 and p.max() <= 1) else (p - p.min()) / (p.max() - p.min())
        r = {"model": name, "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, pp)}
        for t in [0.95, 0.90]:
            sp, pn, se = spec_at_sens(y, p, t); r[f"spec@{int(t*100)}"] = sp; r[f"negfrac@{int(t*100)}"] = pn
        rows.append(r)
    d = pd.DataFrame(rows)
    return {"model": name, **{f"{c}_mean": d[c].mean() for c in d.columns if c != "model"}, **{f"{c}_sd": d[c].std() for c in d.columns if c != "model"}}

def run_cv(model_fn, cols, pool_mask=None, repeats=args.repeats, name=""):
    """OOF predictions on the intermediate cohort; optional extra training pool (rows outside intermediate)."""
    Xi = Xall.loc[is_inter, cols]; yi = yall[is_inter]
    Xp = Xall.loc[pool_mask, cols] if pool_mask is not None else None
    yp = yall[pool_mask] if pool_mask is not None else None
    oofs = []
    for r in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        oof = np.zeros(len(yi))
        for a, v in skf.split(Xi, yi):
            Xa, ya = Xi.iloc[a], yi[a]
            if Xp is not None: Xa = pd.concat([Xa, Xp]); ya = np.concatenate([ya, yp])
            m = model_fn(); m.fit(Xa, ya); oof[v] = m.predict_proba(Xi.iloc[v])[:, 1]
        oofs.append(oof)
    return yi, oofs

results = []; oof_store = {}
t0 = time.time()
for fs in ["F0", "Fc", "F1", "F2"]:
    cols = FS[fs]; models = make_models(cols)
    for mname, model in models.items():
        import copy
        yi, oofs = run_cv(lambda: copy.deepcopy(model), cols, name=f"{fs}/{mname}")
        s = summarize(f"{fs}/{mname}", yi, oofs); s.update({"feature_set": fs, "pool": "intermediate"}); results.append(s)
        oof_store[f"{fs}/{mname}"] = np.mean(oofs, axis=0)
        P(f"{fs:>2} {mname:<16} AUC={s['auc_mean']:.3f}±{s['auc_sd']:.3f}  spec@95={s['spec@95_mean']:.3f}  neg@95={s['negfrac@95_mean']:.3f}  spec@90={s['spec@90_mean']:.3f}  brier={s['brier_mean']:.4f}  ({time.time()-t0:.0f}s)")
# point score
yi = yall[is_inter]; ps = point_score(Xall.loc[is_inter])
s = summarize("F0/PointScore", yi, [ps.values]); s.update({"feature_set": "F0", "pool": "none"}); results.append(s)
P(f"PointScore AUC={s['auc_mean']:.3f} spec@95={s['spec@95_mean']:.3f} spec@90={s['spec@90_mean']:.3f}")

# ----------------------------------------------------------------------------- augmented training pools (F1; LR and LightGBM)
for pool_name, mask in [("+verylow_imaged", is_verylow), ("+verylow+high_imaged", is_verylow | is_high)]:
    cols = FS["F1"]; models = make_models(cols)
    for mname in ["LR_L2", "LR_age_spline", "LightGBM", "RandomForest"]:
        import copy
        yi, oofs = run_cv(lambda: copy.deepcopy(models[mname]), cols, pool_mask=mask)
        s = summarize(f"F1/{mname}{pool_name}", yi, oofs); s.update({"feature_set": "F1", "pool": pool_name}); results.append(s)
        oof_store[f"F1/{mname}{pool_name}"] = np.mean(oofs, axis=0)
        P(f"F1 {mname:<16}{pool_name:<22} AUC={s['auc_mean']:.3f}±{s['auc_sd']:.3f}  spec@95={s['spec@95_mean']:.3f}  neg@95={s['negfrac@95_mean']:.3f}  spec@90={s['spec@90_mean']:.3f}")

res = pd.DataFrame(results).sort_values("auc_mean", ascending=False)
res.to_csv(os.path.join(OUT, "summary_cv.csv"), index=False)
P("\nTOP 10 by CV AUC:\n" + res[["model", "auc_mean", "auc_sd", "spec@95_mean", "negfrac@95_mean", "spec@90_mean", "negfrac@90_mean", "brier_mean"]].head(10).to_string(index=False))

# ----------------------------------------------------------------------------- decision curve (net benefit) for selected models
sel = ["F0/LR_L2", "F1/LR_L2", "F1/LR_age_spline", "F1/LightGBM", "F1/LR_L2+verylow_imaged"] + [c for c in [res.iloc[0]["model"]] if c in oof_store]
sel = list(dict.fromkeys([m for m in sel if m in oof_store]))
pts = np.arange(0.01, 0.16, 0.005); prev = yi.mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pts, [prev - (1 - prev) * pt / (1 - pt) for pt in pts], "k-", lw=1.5, label="CT for all (current practice)")
ax.plot(pts, [0] * len(pts), "k--", lw=1, label="CT for none")
for m in sel:
    ax.plot(pts, [net_benefit(yi, oof_store[m], pt) for pt in pts], lw=1.6, label=m)
ax.set_xlabel("Threshold probability of TBI on CT"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.01, prev + 0.01)
ax.set_title("Decision curve analysis (out-of-fold, intermediate-risk infants < 2 y)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "decision_curve.png"), dpi=200); plt.close()
dca = pd.DataFrame({"pt": pts, "CT_all": [prev - (1 - prev) * pt / (1 - pt) for pt in pts], **{m: [net_benefit(yi, oof_store[m], pt) for pt in pts] for m in sel}})
dca.to_csv(os.path.join(OUT, "decision_curve.csv"), index=False)

# ----------------------------------------------------------------------------- fixed 80/20 split (seed 42) for top models
Xi = Xall.loc[is_inter]; yi = yall[is_inter]
idx_tr, idx_te = train_test_split(np.arange(len(yi)), test_size=0.2, random_state=SEED, stratify=yi)
split_rows = []
top_models = [m for m in res["model"].head(8) if "+" not in m and m in oof_store]
for key in top_models + ["F0/LR_L2"]:
    fs, mname = key.split("/")
    cols = FS[fs]; model = make_models(cols)[mname]
    import copy
    m = copy.deepcopy(model).fit(Xi.iloc[idx_tr][cols], yi[idx_tr]); p = m.predict_proba(Xi.iloc[idx_te][cols])[:, 1]
    ptr = m.predict_proba(Xi.iloc[idx_tr][cols])[:, 1]
    # threshold from training OOF (5-fold) at 95% sensitivity, applied to test
    oof = np.zeros(len(idx_tr))
    for a, v in StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx_tr, yi[idx_tr]):
        mm = copy.deepcopy(model).fit(Xi.iloc[idx_tr[a]][cols], yi[idx_tr[a]]); oof[v] = mm.predict_proba(Xi.iloc[idx_tr[v]][cols])[:, 1]
    pos = np.sort(oof[yi[idx_tr] == 1]); thr = pos[int(np.floor(0.05 * len(pos)))]
    pred = p >= thr; yt = yi[idx_te]
    sp95, pn95, _ = spec_at_sens(yt, p, 0.95); sp90, pn90, _ = spec_at_sens(yt, p, 0.90)
    split_rows.append({"model": key, "test_auc": roc_auc_score(yt, p), "thr_oof95": thr, "test_sens@thr": pred[yt == 1].mean(), "test_spec@thr": (~pred[yt == 0]).mean(),
                       "test_negfrac@thr": (~pred).mean(), "test_spec@95": sp95, "test_negfrac@95": pn95, "test_spec@90": sp90, "test_negfrac@90": pn90})
sp = pd.DataFrame(split_rows); sp.to_csv(os.path.join(OUT, "fixed_split_test.csv"), index=False)
P("\nFixed 80/20 split (test n=%d):\n" % len(idx_te) + sp.round(3).to_string(index=False))
P("DONE in %.0fs" % (time.time() - t0)); log.close()
