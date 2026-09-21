# =============================================================================
#  CROSS-COUNTRY ANALYSIS OF THE MISSING-AWARE FAILURE MODEL
#  Zombie Firms lab (IMT Lucca) -- one self-contained script
# =============================================================================
#  Answers, in order:
#    PART 1  Data checks        What does "missing" mean in each country? (size-band firms, sign flip,
#                               filing patterns, ownership, sector mix)
#    PART 2  Own-country models 5-fold out-of-fold baseline for each country (the benchmark everything
#                               else is compared with)
#    PART 3  Leave one out      Homework: train on 3 countries, test on the 4th
#    PART 4  Transfer matrix    Train on one country, test on each other country (4x4) + cutoff transfer
#    PART 5  Harmonization      Can the countries be made comparable? H0 raw, H1 drop IT size-band firms,
#                               H2 within-country ranks, H3 covariate reweighting, H4 filers only
#    PART 6  Adaptation         Cutoff rules in the new country, few-shot adaptation with 5% labelled
#                               target firms, label budget (1-20%)
#    PART 7  Export             All tables to CSV, figures to PNG
#
#  How to run
#    * Colab: paste the whole file into ONE cell (or upload it and run `%run cross_country_analysis.py`).
#      It mounts Google Drive and finds data_lagged_10_07.csv anywhere under MyDrive.
#    * Locally: `python cross_country_analysis.py --data path/to/data_lagged_10_07.csv`
#  Runtime (2-core CPU, full data): QUICK = False ~12-20 min; QUICK = True ~4-6 min.
#  Everything is seeded; results match the report (zombie_firms_lab_report.pdf, sections 6-9).
#
#  Requirements: numpy, pandas, scikit-learn, xgboost, matplotlib (all preinstalled in Colab).
# =============================================================================
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------
QUICK = False            # True = fewer trees and folds, for a fast check that everything runs
RUN = {                  # switch parts on/off (later parts reuse results of earlier ones when available)
    "checks": True, "own_country": True, "leave_one_out": True, "transfer_matrix": True,
    "harmonization": True, "adaptation": True,
}
N_TREES = 150 if QUICK else 500
N_FOLDS = 3 if QUICK else 5
SEED = 42
FEW_SHOT_SHARE = 0.05                       # labelled share of the target country in PART 6
BUDGET_SHARES = (0.01, 0.02, 0.05, 0.10, 0.20)
COUNTRIES = ["IT", "FR", "PT", "ES"]
BAND_VALUES = [250_000, 750_000, 1_500_000, 3_500_000]  # mid-points of 0-0.5M, 0.5-1M, 1-2M, 2-5M EUR

NUM = ["Number_of_patents", "Number_of_trademarks", "consdummy", "tfp_acf", "fin_rev", "int_paid", "ebitda",
       "cash_flow", "depr", "revenue", "total_assets", "long_term_debt", "employees", "added_value", "materials",
       "wage_bill", "loans", "int_fixed_assets", "fixed_assets", "current_liabilities", "liquidity_ratio",
       "solvency_ratio", "current_assets", "fin_expenses", "net_income", "fin_cons100", "inv", "real_SA",
       "shareholders_funds", "NEG_VA", "ICR_failure", "profitability", "misallocated_fixed", "interest_diff"]
CAT = ["nace_2", "control"]   # 'iso' is deliberately NOT a feature: constant within a country and unseen
                              # for a held-out country


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def in_notebook():
    return "ipykernel" in sys.modules


def finish_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.show() if in_notebook() else plt.close()


def locate_data():
    """Command-line --data > Colab Drive search > current folder / Dati_CSV subfolders."""
    if not in_notebook():
        ap = argparse.ArgumentParser()
        ap.add_argument("--data", default=None)
        ap.add_argument("--out", default="cross_country_outputs")
        args, _ = ap.parse_known_args()
        if args.data:
            return Path(args.data), Path(args.out)
    roots = [Path.cwd()]
    if in_colab():
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        roots = [Path("/content/drive/MyDrive"), Path("/content")]
    for root in roots:
        hits = sorted(root.rglob("data_lagged_10_07.csv"))
        if hits:
            out = Path("/content/cross_country_outputs") if in_colab() else Path("cross_country_outputs")
            return hits[0], out
    raise FileNotFoundError("data_lagged_10_07.csv not found. Pass --data PATH or put it under this folder / MyDrive.")


try:
    LAGGED_FILE, OUT_DIR = Path(LAGGED_FILE), Path(OUTPUT_DIR) / "cross_country"  # reuse notebook variables if defined
except NameError:
    LAGGED_FILE, OUT_DIR = locate_data()
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from IPython.display import display as _display
    def show(df, digits=3):
        _display(df.round(digits) if isinstance(df, pd.DataFrame) else df)
except ImportError:
    def show(df, digits=3):
        print(df.round(digits).to_string() if isinstance(df, pd.DataFrame) else df)


def header(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90, flush=True)


print(f"Data: {LAGGED_FILE}\nOutputs: {OUT_DIR}\nQUICK = {QUICK} (trees = {N_TREES}, folds = {N_FOLDS})")
T0 = time.time()

# -----------------------------------------------------------------------------
# DATA AND SHARED HELPERS
# -----------------------------------------------------------------------------
raw = pd.read_csv(LAGGED_FILE, usecols=["iso", "failure"] + CAT + NUM, low_memory=False)
raw = raw.dropna(subset=["failure"]).reset_index(drop=True)
raw["failure"] = raw["failure"].astype(int)
raw = raw[raw["iso"].isin(COUNTRIES)].reset_index(drop=True)
Y = raw["failure"].to_numpy()
ISO = raw["iso"].to_numpy()
# One fixed dummy set for all countries -> no unseen categories when moving between countries
CATS = pd.get_dummies(raw[CAT].astype("string").fillna("__MISSING__"), dtype=float).to_numpy()
BANDED = (raw["iso"].eq("IT") & raw["total_assets"].isna() & raw["revenue"].isin(BAND_VALUES)).to_numpy()
# Continuous variables only are ranked in H2: ranking a binary variable would give the SAME value a
# DIFFERENT rank in each country (a tie's rank depends on the country's prevalence).
CONTINUOUS = [c for c in NUM if raw[c].nunique() > 50]


def xgb(pos_weight, n=N_TREES):
    return XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=5, reg_lambda=2.0, eval_metric="logloss", scale_pos_weight=pos_weight,
                         tree_method="hist", random_state=SEED, n_jobs=-1)


def pos_w(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def bacc_cutoff(y, p):
    """Cutoff maximising balanced accuracy over 199 quantile candidates (the notebook's rule)."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, p >= t) for t in grid]))])


def scores(y, p, cutoff=None):
    out = {"n": len(y), "fail_rate": y.mean(), "ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}
    out["lift"] = out["PR_AUC"] / out["fail_rate"]
    if cutoff is not None:  # a cutoff chosen elsewhere, applied as is
        out["share_flagged"] = (p >= cutoff).mean()
        out["BACC_at_cutoff"] = balanced_accuracy_score(y, p >= cutoff)
        out["recall_at_cutoff"] = (p[y == 1] >= cutoff).mean()
    return out


DESIGNS = {
    "H0 raw": "data as delivered",
    "H1 drop IT size-band firms": "population harmonization",
    "H2 H1 + within-country ranks": "feature harmonization (continuous variables)",
    "H3 H1 + covariate reweighting": "source firms reweighted to resemble the target",
    "H4 H2, filers only": "diagnostic: firms with a balance sheet only",
}


def design(name):
    """Feature matrix and sample mask for a harmonization design."""
    keep = np.ones(len(raw), bool)
    if name != "H0 raw":
        keep &= ~BANDED
    X = raw[NUM].astype(float)
    if name in ("H2 H1 + within-country ranks", "H4 H2, filers only"):
        X.loc[keep, CONTINUOUS] = raw.loc[keep, CONTINUOUS].groupby(raw.loc[keep, "iso"]).rank(pct=True)
    if name == "H4 H2, filers only":
        keep &= raw["total_assets"].notna().to_numpy()
    return np.hstack([X.to_numpy(float), CATS]), keep


def own_country_cv(X, keep):
    """5-fold out-of-fold performance within each country."""
    rows = []
    for c in COUNTRIES:
        m = keep & (ISO == c)
        Xc, yc = X[m], Y[m]
        p = np.zeros(len(yc))
        for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xc, yc):
            p[te] = xgb(pos_w(yc[tr])).fit(Xc[tr], yc[tr]).predict_proba(Xc[te])[:, 1]
        rows.append({"country": c, **scores(yc, p)})
    return pd.DataFrame(rows).set_index("country")


SOURCE_MODELS = {}   # (design, country) -> (fitted model, cutoff chosen on that country)


def source_model(name, X, keep, c):
    if (name, c) not in SOURCE_MODELS:
        m = keep & (ISO == c)
        mdl = xgb(pos_w(Y[m])).fit(X[m], Y[m])
        SOURCE_MODELS[(name, c)] = (mdl, bacc_cutoff(Y[m], mdl.predict_proba(X[m])[:, 1]))
    return SOURCE_MODELS[(name, c)]


def covariate_weights(X_src, X_tgt):
    """Importance weights p(target|x)/p(source|x) from a domain classifier (Shimodaira 2000)."""
    Z = np.vstack([X_src, X_tgt])
    d = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
    clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, tree_method="hist",
                        random_state=SEED, n_jobs=-1).fit(Z, d)
    p = np.clip(clf.predict_proba(X_src)[:, 1], 1e-3, 1 - 1e-3)
    w = p / (1 - p) * len(X_src) / len(X_tgt)
    w = np.clip(w, None, np.quantile(w, 0.99))
    # in-sample AUC of the domain classifier: 0.5 = countries indistinguishable, 1.0 = no overlap
    return w / w.mean(), roc_auc_score(d, clf.predict_proba(Z)[:, 1])


RESULTS = {}

# -----------------------------------------------------------------------------
# PART 1. DATA CHECKS: what does "missing" mean in each country?
# -----------------------------------------------------------------------------
if RUN["checks"]:
    header("PART 1. Data checks")
    chk = raw.assign(has_bs=raw["total_assets"].notna(), has_is=raw["ebitda"].notna())
    print("1a) Most frequent revenue values (a genuine revenue distribution has no spikes):")
    for c in COUNTRIES:
        print(f"   {c}:", chk.loc[chk.iso == c, "revenue"].value_counts().head(4).to_dict())
    it = chk.iso.eq("IT")
    print(f"\n1b) Italian size-band firms (no balance sheet, revenue exactly {BAND_VALUES}):"
          f" {BANDED.sum():,} = {BANDED[it].mean():.1%} of IT; failure rate {Y[BANDED].mean():.2%}"
          f" vs {Y[it.to_numpy() & ~BANDED].mean():.2%} for other IT firms")
    print("\n1c) Failure rate with vs without a balance sheet (sign flips in IT):")
    t = chk.pivot_table(index="iso", columns="has_bs", values="failure", aggfunc="mean").loc[COUNTRIES]
    t.columns = ["no balance sheet", "balance sheet present"]
    show(t[["balance sheet present", "no balance sheet"]], 4)
    RESULTS["checks_signflip"] = t
    print("\n1d) Filing patterns: share of firms and failure rate")
    chk["pattern"] = np.select([chk.has_bs & chk.has_is, chk.has_bs], ["full accounts", "balance sheet only"], "no accounts")
    pat = chk.groupby(["iso", "pattern"])["failure"].agg(firms="size", fail_rate="mean")
    pat["share"] = pat["firms"] / pat.groupby(level=0)["firms"].transform("sum")
    show(pat[["share", "fail_rate"]].unstack(0))
    RESULTS["checks_patterns"] = pat.reset_index()
    print("\n1e) Share of firms with no missing predictor (notebook section 5.1) and missing rate of key variables")
    miss = chk.groupby("iso")[["employees", "revenue", "total_assets", "ebitda", "tfp_acf", "int_paid"]].apply(lambda g: g.isna().mean())
    miss.insert(0, "complete_case_share", chk.groupby("iso").apply(lambda g: g[NUM + CAT].notna().all(axis=1).mean()))
    show(miss.loc[COUNTRIES])
    RESULTS["checks_missing"] = miss
    print("\n1f) Failure rate by ownership type (same pattern in every country)")
    show(chk.pivot_table(index="control", columns="iso", values="failure", aggfunc="mean")[COUNTRIES], 4)
    print("\n1g) Sector mix: dissimilarity vs the other three countries (0 = identical) and failure rate implied by sector mix")
    shares = pd.crosstab(chk["nace_2"], chk["iso"], normalize="columns")
    sector_fail = chk.groupby("nace_2")["failure"].mean()
    rows = []
    for c in COUNTRIES:
        others = chk.loc[~chk.iso.eq(c), "nace_2"].value_counts(normalize=True).reindex(shares.index).fillna(0)
        rows.append({"country": c, "dissimilarity": 0.5 * (shares[c] - others).abs().sum(),
                     "actual_fail_rate": chk.loc[chk.iso.eq(c), "failure"].mean(),
                     "implied_by_sector_mix": (shares[c] * sector_fail).sum()})
    RESULTS["checks_sector"] = pd.DataFrame(rows).set_index("country")
    show(RESULTS["checks_sector"])

# -----------------------------------------------------------------------------
# PART 2. OWN-COUNTRY BASELINES (5-fold out-of-fold)
# -----------------------------------------------------------------------------
OWN = {}
if RUN["own_country"]:
    header("PART 2. Own-country models (the benchmark)")
    for name in ("H0 raw", "H1 drop IT size-band firms"):
        X, keep = design(name)
        OWN[name] = own_country_cv(X, keep)
        print(f"\n{name}:")
        show(OWN[name])
    # ranks are monotone within a country and reweighting only applies across countries -> same baseline
    OWN["H2 H1 + within-country ranks"] = OWN["H3 H1 + covariate reweighting"] = OWN["H1 drop IT size-band firms"]
    RESULTS["own_country"] = pd.concat({k: v for k, v in OWN.items()}, names=["design"]).reset_index()
    print(f"[{(time.time() - T0) / 60:.1f} min]")

# -----------------------------------------------------------------------------
# PART 3. LEAVE ONE COUNTRY OUT (homework)
# -----------------------------------------------------------------------------
if RUN["leave_one_out"]:
    header("PART 3. Leave one country out: train on three countries, test on the fourth")
    rows = []
    for name in ("H0 raw", "H1 drop IT size-band firms"):
        X, keep = design(name)
        for c in COUNTRIES:
            tr, te = keep & (ISO != c), keep & (ISO == c)
            mdl = xgb(pos_w(Y[tr])).fit(X[tr], Y[tr])
            cut = bacc_cutoff(Y[tr], mdl.predict_proba(X[tr])[:, 1])
            r = {"design": name, "test": c, **scores(Y[te], mdl.predict_proba(X[te])[:, 1], cut)}
            if name in OWN:
                r["PR_AUC_own"] = OWN[name].loc[c, "PR_AUC"]
                r["ROC_AUC_own"] = OWN[name].loc[c, "ROC_AUC"]
                r["PR_gap"] = r["PR_AUC"] - r["PR_AUC_own"]
            rows.append(r)
            print(f"   {name:28s} test {c}: PR AUC {r['PR_AUC']:.3f}" + (f" (own {r['PR_AUC_own']:.3f})" if "PR_AUC_own" in r else ""))
    RESULTS["leave_one_out"] = pd.DataFrame(rows)
    show(RESULTS["leave_one_out"].set_index(["design", "test"])[
        [c for c in ["ROC_AUC_own", "ROC_AUC", "PR_AUC_own", "PR_AUC", "PR_gap", "share_flagged", "BACC_at_cutoff"]
         if c in RESULTS["leave_one_out"]]])
    print(f"[{(time.time() - T0) / 60:.1f} min]")

# -----------------------------------------------------------------------------
# PART 4 + 5. TRANSFER MATRIX AND HARMONIZATION (train on one country, test on each other)
# -----------------------------------------------------------------------------
transfer_rows = []
designs_to_run = []
if RUN["transfer_matrix"]:
    designs_to_run += ["H0 raw", "H1 drop IT size-band firms"]
if RUN["harmonization"]:
    designs_to_run += ["H2 H1 + within-country ranks", "H3 H1 + covariate reweighting", "H4 H2, filers only"]
    if "H1 drop IT size-band firms" not in designs_to_run:
        designs_to_run.insert(0, "H1 drop IT size-band firms")
if designs_to_run:
    header("PART 4-5. Transfer matrix and harmonization")
for name in designs_to_run:
    X, keep = design(name)
    if name == "H4 H2, filers only":
        OWN[name] = own_country_cv(X, keep)  # the filers-only sample needs its own baseline
    for a in COUNTRIES:
        src = keep & (ISO == a)
        if name != "H3 H1 + covariate reweighting":
            mdl, cut = source_model(name, X, keep, a)
        for b in COUNTRIES:
            if a == b:
                continue
            tgt = keep & (ISO == b)
            dom_auc = np.nan
            if name == "H3 H1 + covariate reweighting":
                w, dom_auc = covariate_weights(X[src], X[tgt])
                mdl = xgb(pos_w(Y[src])).fit(X[src], Y[src], sample_weight=w)
                cut = bacc_cutoff(Y[src], mdl.predict_proba(X[src])[:, 1])
            r = {"design": name, "train": a, "test": b, **scores(Y[tgt], mdl.predict_proba(X[tgt])[:, 1], cut),
                 "domain_AUC": dom_auc}
            if name in OWN:
                r["PR_AUC_own"] = OWN[name].loc[b, "PR_AUC"]
                r["PR_gap"] = r["PR_AUC"] - r["PR_AUC_own"]
                r["PR_share_kept"] = r["PR_AUC"] / r["PR_AUC_own"]
            transfer_rows.append(r)
    print(f"   {name}: done [{(time.time() - T0) / 60:.1f} min]", flush=True)

if transfer_rows:
    TR = pd.DataFrame(transfer_rows)
    RESULTS["transfer"] = TR
    for name in designs_to_run:
        mat = TR[TR.design == name].pivot(index="train", columns="test", values="PR_AUC")
        if name in OWN:
            for c in COUNTRIES:
                mat.loc[c, c] = OWN[name].loc[c, "PR_AUC"]
        print(f"\nPR AUC, rows = trained on, columns = tested on (diagonal = own country) -- {name}")
        show(mat.loc[COUNTRIES, COUNTRIES])
        if name in ("H0 raw", "H1 drop IT size-band firms"):
            print("Share of target firms flagged with the TRAINING country's cutoff:")
            show(TR[TR.design == name].pivot(index="train", columns="test", values="share_flagged").loc[COUNTRIES, COUNTRIES], 2)
    if "PR_gap" in TR:
        summary = (TR.groupby("design", sort=False)
                   .agg(mean_PR_AUC=("PR_AUC", "mean"), mean_ROC_AUC=("ROC_AUC", "mean"), mean_PR_gap=("PR_gap", "mean"),
                        worst_PR_gap=("PR_gap", "min"), mean_share_kept=("PR_share_kept", "mean"),
                        mean_domain_AUC=("domain_AUC", "mean")))
        print("\nHarmonization summary (mean over the 12 country pairs):")
        show(summary)
        RESULTS["harmonization_summary"] = summary.reset_index()

# -----------------------------------------------------------------------------
# PART 6. ADAPTATION: cutoffs, few-shot, label budget (on the harmonized population H1)
# -----------------------------------------------------------------------------
if RUN["adaptation"]:
    header("PART 6. Adaptation to the target country (design H1)")
    name = "H1 drop IT size-band firms"
    X, keep = design(name)
    cut_rows, few_rows, budget_rows = [], [], []
    for a in COUNTRIES:
        src = keep & (ISO == a)
        m_src, t_src = source_model(name, X, keep, a)
        flag_src = (m_src.predict_proba(X[src])[:, 1] >= t_src).mean()
        for b in COUNTRIES:
            if a == b:
                continue
            idx = np.where(keep & (ISO == b))[0]
            # --- (i) cutoff rules and (ii) few-shot models: 5% labelled, 95% evaluation ---
            lab, ev = train_test_split(idx, train_size=FEW_SHOT_SHARE, stratify=Y[idx], random_state=SEED)
            p_lab, p_ev = m_src.predict_proba(X[lab])[:, 1], m_src.predict_proba(X[ev])[:, 1]
            pi_s, pi_t = Y[src].mean(), Y[idx].mean()   # target failure rate assumed known (official statistics)
            rules = {
                "C0 source cutoff": t_src,
                "C1 same share flagged as in source": np.quantile(p_ev, 1 - flag_src),
                "C2 share scaled by failure-rate ratio": np.quantile(p_ev, 1 - min(flag_src * pi_t / pi_s, 0.99)),
                "C3 cutoff tuned on 5% labelled target": bacc_cutoff(Y[lab], p_lab),
                "Oracle: best cutoff on target": bacc_cutoff(Y[ev], p_ev),
            }
            for rule, t in rules.items():
                cut_rows.append({"train": a, "test": b, "rule": rule, "cutoff": t, "share_flagged": (p_ev >= t).mean(),
                                 "recall": (p_ev[Y[ev] == 1] >= t).mean(), "BACC": balanced_accuracy_score(Y[ev], p_ev >= t)})
            variants = {"transferred model only": p_ev,
                        "target-only model (5%)": xgb(pos_w(Y[lab]), 300).fit(X[lab], Y[lab]).predict_proba(X[ev])[:, 1]}
            Xp, yp = np.vstack([X[src], X[lab]]), np.r_[Y[src], Y[lab]]
            wp = np.r_[np.ones(src.sum()), np.full(len(lab), src.sum() / len(lab))]
            variants["pooled source + 5% target (up-weighted)"] = (xgb(pos_w(yp)).fit(Xp, yp, sample_weight=wp)
                                                                   .predict_proba(X[ev])[:, 1])
            warm = xgb(pos_w(Y[lab]), 150)
            warm.fit(X[lab], Y[lab], xgb_model=m_src.get_booster())   # continue boosting on target labels
            variants["warm start: transferred + 150 trees on 5%"] = warm.predict_proba(X[ev])[:, 1]
            for v, p in variants.items():
                few_rows.append({"train": a, "test": b, "variant": v, "n_labelled": len(lab),
                                 "failures_labelled": int(Y[lab].sum()), "PR_AUC": average_precision_score(Y[ev], p),
                                 "ROC_AUC": roc_auc_score(Y[ev], p)})
            # --- (iii) label budget: fixed 80% evaluation set, 1-20% labelled ---
            pool, ev2 = train_test_split(idx, train_size=0.20, stratify=Y[idx], random_state=SEED)
            budget_rows.append({"train": a, "test": b, "label_share": 0.0, "n_labelled": 0, "failures_labelled": 0,
                                "variant": "warm start", "PR_AUC": average_precision_score(Y[ev2], m_src.predict_proba(X[ev2])[:, 1])})
            for share in BUDGET_SHARES:
                labs = pool if share >= 0.20 else train_test_split(pool, train_size=share / 0.20, stratify=Y[pool],
                                                                    random_state=SEED)[0]
                w_m = xgb(pos_w(Y[labs]), 150)
                w_m.fit(X[labs], Y[labs], xgb_model=m_src.get_booster())
                o_m = xgb(pos_w(Y[labs]), 300).fit(X[labs], Y[labs])
                for v, mdl in (("warm start", w_m), ("target-only", o_m)):
                    budget_rows.append({"train": a, "test": b, "label_share": share, "n_labelled": len(labs),
                                        "failures_labelled": int(Y[labs].sum()), "variant": v,
                                        "PR_AUC": average_precision_score(Y[ev2], mdl.predict_proba(X[ev2])[:, 1])})
        print(f"   source {a} done [{(time.time() - T0) / 60:.1f} min]", flush=True)
    RESULTS["cutoffs"], RESULTS["few_shot"], RESULTS["label_budget"] = map(pd.DataFrame, (cut_rows, few_rows, budget_rows))
    print("\n(i) Cutoff rules -- mean over the 12 pairs (evaluated on 95% of the target):")
    show(RESULTS["cutoffs"].groupby("rule", sort=False)[["BACC", "share_flagged", "recall"]].mean())
    print("\n(ii) Few-shot adaptation with 5% labelled target firms -- mean over the 12 pairs:")
    show(RESULTS["few_shot"].groupby("variant", sort=False)[["PR_AUC", "ROC_AUC"]].mean())
    print("\n(iii) Label budget -- PR AUC by target country (mean over the 3 source countries):")
    show(RESULTS["label_budget"].pivot_table(index=["test", "label_share"], columns="variant", values="PR_AUC"))

# -----------------------------------------------------------------------------
# PART 7. EXPORT: CSV tables and figures
# -----------------------------------------------------------------------------
header("PART 7. Export")
for key, df in RESULTS.items():
    df.to_csv(OUT_DIR / f"{key}.csv", index=not isinstance(df.index, pd.RangeIndex))
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "font.size": 9, "axes.titleweight": "bold"})
if "checks_signflip" in RESULTS:
    t = RESULTS["checks_signflip"] * 100
    fig, ax = plt.subplots(figsize=(6.2, 2.8)); x = np.arange(len(t)); w = 0.36
    ax.bar(x - w / 2, t["balance sheet present"], w, color=BLUE, label="Balance sheet present", edgecolor="white")
    ax.bar(x + w / 2, t["no balance sheet"], w, color=ORANGE, label="Balance sheet missing", edgecolor="white")
    ax.set_xticks(x, t.index); ax.set_ylabel("Failure rate (%)"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False); ax.set_title("Missing accounts signal risk everywhere except Italy", loc="left")
    finish_figure(OUT_DIR / "fig_missingness_sign.png")
if transfer_rows:
    shown = [d for d in ("H0 raw", "H1 drop IT size-band firms") if d in designs_to_run and d in OWN]
    if shown:
        fig, axs = plt.subplots(1, len(shown), figsize=(3.4 * len(shown), 3.0), squeeze=False)
        for ax, name in zip(axs[0], shown):
            mat = TR[TR.design == name].pivot(index="train", columns="test", values="PR_AUC")
            for c in COUNTRIES:
                mat.loc[c, c] = OWN[name].loc[c, "PR_AUC"]
            mat = mat.loc[COUNTRIES, COUNTRIES]
            ax.imshow(mat.values, cmap="Blues", vmin=0.4, vmax=0.8); ax.grid(False)
            for i in range(4):
                for j in range(4):
                    v = mat.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.66 else "black", fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(4), COUNTRIES); ax.set_yticks(range(4), COUNTRIES)
            ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on"); ax.set_title(name, loc="left", fontsize=8)
        finish_figure(OUT_DIR / "fig_transfer_matrix.png")
if "label_budget" in RESULTS:
    g = RESULTS["label_budget"].groupby(["test", "variant", "label_share"])["PR_AUC"].mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(9, 2.6), sharey=True)
    for ax, c in zip(axs, COUNTRIES):
        for v, col, lab in (("warm start", BLUE, "Transferred + target labels"), ("target-only", ORANGE, "Target labels only")):
            q = g[(g.test == c) & (g.variant == v)].sort_values("label_share")
            ax.plot(q.label_share * 100, q.PR_AUC, color=col, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(c, loc="left"); ax.set_xlabel("% of target labelled")
    axs[0].set_ylabel("PR AUC"); axs[0].legend(frameon=False, fontsize=7)
    finish_figure(OUT_DIR / "fig_label_budget.png")

print(f"\nSaved {len(RESULTS)} tables and the figures to {OUT_DIR}")
print(f"Total runtime: {(time.time() - T0) / 60:.1f} min")
