# =============================================================================
# 15. Making the model compatible across countries (harmonization + adaptation)
# Paste as ONE code cell after section 14. Self-contained: needs only LAGGED_FILE and
# OUTPUT_DIR from section 2. Runtime on a Colab CPU: ~40-60 min with N_TREES = 500
# (set N_TREES = 200 for a ~20 min check).
#
# Question: a model trained on country A is used on country B. What has to change so
# that it works there? We add fixes one at a time and re-run the full 4x4 transfer
# matrix after each one:
#   H0  raw data (as delivered)                                      -> baseline
#   H1  population harmonization: drop IT firms that only report a revenue size band
#   H2  H1 + feature harmonization: within-country percentile ranks of continuous variables
#   H3  H1 + covariate-shift reweighting (source firms weighted to look like the target)
#   H4  H2 restricted to firms that filed a balance sheet (diagnostic: removes the
#       country-specific meaning of "missing")
# Then, on H1 (the best-performing harmonized design):
#   Cutoffs  C0 source cutoff, C1 same flag-share as source, C2 flag-share scaled by the
#            target's known failure rate, C3 cutoff tuned on 5% labelled target firms,
#            Oracle = best possible cutoff on the target (upper bound)
#   Few-shot with 5% labelled target firms: target-only model vs source model vs
#            pooled (target up-weighted) vs warm-start (extra trees on target data)
#   Label budget: warm start vs target-only with 1%, 2%, 5%, 10%, 20% labelled target firms
# =============================================================================
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

N_TREES = 500
SEED = 42
FEW_SHOT_SHARE = 0.05
COUNTRIES = ["IT", "FR", "PT", "ES"]
NUM = ["Number_of_patents", "Number_of_trademarks", "consdummy", "tfp_acf", "fin_rev", "int_paid", "ebitda",
       "cash_flow", "depr", "revenue", "total_assets", "long_term_debt", "employees", "added_value", "materials",
       "wage_bill", "loans", "int_fixed_assets", "fixed_assets", "current_liabilities", "liquidity_ratio",
       "solvency_ratio", "current_assets", "fin_expenses", "net_income", "fin_cons100", "inv", "real_SA",
       "shareholders_funds", "NEG_VA", "ICR_failure", "profitability", "misallocated_fixed", "interest_diff"]

raw = pd.read_csv(LAGGED_FILE, usecols=["iso", "failure", "nace_2", "control"] + NUM, low_memory=False)
raw = raw.dropna(subset=["failure"]).reset_index(drop=True)
raw["failure"] = raw["failure"].astype(int)
# categorical columns -> fixed dummy set shared by all countries (no unseen categories)
cats = pd.get_dummies(raw[["nace_2", "control"]].astype("string").fillna("__MISSING__"), dtype=float)
banded = (raw["iso"].eq("IT") & raw["total_assets"].isna()
          & raw["revenue"].isin([250_000, 750_000, 1_500_000, 3_500_000]))


# Only continuous variables are ranked. Ranking a binary/count variable (NEG_VA, ICR_failure, patents...)
# would give the SAME value (e.g. NEG_VA = 0) a DIFFERENT rank in each country, because the rank of a tie
# depends on the country's prevalence -- that destroys comparability instead of creating it (a first
# version of this cell ranked everything; the domain classifier then separated countries perfectly).
CONTINUOUS = [c for c in NUM if raw[c].nunique() > 50]


def within_country_ranks(df):
    # Percentile rank of each continuous variable inside its own country (NaN stays NaN). A firm at the
    # 90th percentile of leverage in Portugal and in France now gets the same value, whatever the
    # country's price level, accounting conventions or size distribution.
    return df[CONTINUOUS].groupby(raw.loc[df.index, "iso"]).rank(pct=True)


def xgb(pos_weight, n=N_TREES):
    return XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=5, reg_lambda=2.0, eval_metric="logloss", scale_pos_weight=pos_weight,
                         tree_method="hist", random_state=SEED, n_jobs=-1)


def pos_w(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def bacc_cutoff(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, p >= t) for t in grid]))])


def covariate_weights(X_src, X_tgt):
    # Domain classifier: P(firm belongs to target | x). Weight source firms by the odds, so the
    # weighted source sample has the target's distribution of x (Shimodaira 2000; Sugiyama et al. 2007).
    Z = np.vstack([X_src, X_tgt])
    d = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
    clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, tree_method="hist",
                        random_state=SEED, n_jobs=-1).fit(Z, d)
    p = np.clip(clf.predict_proba(X_src)[:, 1], 1e-3, 1 - 1e-3)
    w = p / (1 - p) * len(X_src) / len(X_tgt)
    w = np.clip(w, None, np.quantile(w, 0.99))  # trim extreme weights (variance control)
    # in-sample AUC of the domain classifier: 0.5 = countries indistinguishable, 1.0 = no overlap
    return w / w.mean(), roc_auc_score(d, clf.predict_proba(Z)[:, 1])


def design(setting):
    """Return (X, y, iso, keep-mask) for a harmonization setting."""
    keep = np.ones(len(raw), bool)
    if setting != "H0 raw":
        keep &= ~banded.to_numpy()
    X = raw[NUM].astype(float)
    if setting in ("H2 + within-country ranks", "H4 filers only (diagnostic)"):
        X.loc[keep, CONTINUOUS] = within_country_ranks(raw.loc[keep])
    if setting == "H4 filers only (diagnostic)":
        keep &= raw["total_assets"].notna().to_numpy()
    X = np.hstack([X.to_numpy(float), cats.to_numpy()])
    return X, raw["failure"].to_numpy(), raw["iso"].to_numpy(), keep


SETTINGS = ["H0 raw", "H1 drop IT size-band firms", "H2 + within-country ranks",
            "H3 H1 + covariate reweighting", "H4 filers only (diagnostic)"]
transfer_rows, within_rows = [], []
for setting in SETTINGS:
    X, y, iso, keep = design(setting)
    t0 = time.time()
    # Diagonal: within-country 5-fold CV. Ranks are monotone within a country, so H2/H3 = H1 there.
    if setting in ("H0 raw", "H1 drop IT size-band firms", "H4 filers only (diagnostic)"):
        for c in COUNTRIES:
            m = keep & (iso == c)
            Xc, yc = X[m], y[m]
            p = np.zeros(len(yc))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc, yc):
                p[te] = xgb(pos_w(yc[tr])).fit(Xc[tr], yc[tr]).predict_proba(Xc[te])[:, 1]
            within_rows.append({"setting": setting, "country": c, "n": int(m.sum()), "fail_rate": yc.mean(),
                                "ROC_AUC": roc_auc_score(yc, p), "PR_AUC": average_precision_score(yc, p)})
    for a in COUNTRIES:
        src = keep & (iso == a)
        base_model = None if setting == "H3 H1 + covariate reweighting" else xgb(pos_w(y[src])).fit(X[src], y[src])
        for b in COUNTRIES:
            if a == b:
                continue
            tgt = keep & (iso == b)
            if setting == "H3 H1 + covariate reweighting":
                w, dom_auc = covariate_weights(X[src], X[tgt])
                model = xgb(pos_w(y[src])).fit(X[src], y[src], sample_weight=w)
            else:
                model, dom_auc = base_model, np.nan
            p = model.predict_proba(X[tgt])[:, 1]
            transfer_rows.append({"setting": setting, "train": a, "test": b, "n_test": int(tgt.sum()),
                                  "fail_rate": y[tgt].mean(), "ROC_AUC": roc_auc_score(y[tgt], p),
                                  "PR_AUC": average_precision_score(y[tgt], p), "domain_AUC": dom_auc})
    print(f"{setting}: {time.time() - t0:.0f}s", flush=True)

within = pd.DataFrame(within_rows)
transfer = pd.DataFrame(transfer_rows)
for s in ("H2 + within-country ranks", "H3 H1 + covariate reweighting"):  # same diagonal as H1
    within = pd.concat([within, within[within.setting == "H1 drop IT size-band firms"].assign(setting=s)])
transfer = transfer.merge(within[["setting", "country", "PR_AUC", "ROC_AUC"]].rename(
    columns={"country": "test", "PR_AUC": "PR_AUC_own", "ROC_AUC": "ROC_AUC_own"}), on=["setting", "test"])
transfer["PR_gap"] = transfer["PR_AUC"] - transfer["PR_AUC_own"]
transfer["PR_retained"] = transfer["PR_AUC"] / transfer["PR_AUC_own"]

summary = (transfer.groupby("setting", sort=False)
           .agg(mean_PR_AUC=("PR_AUC", "mean"), mean_PR_gap=("PR_gap", "mean"), worst_PR_gap=("PR_gap", "min"),
                mean_share_retained=("PR_retained", "mean"), mean_ROC_AUC=("ROC_AUC", "mean")).reset_index())
display(summary.round(3))
for s in SETTINGS:
    mat = transfer[transfer.setting == s].pivot(index="train", columns="test", values="PR_AUC")
    own = within[within.setting == s].set_index("country")["PR_AUC"]
    for c in COUNTRIES:
        mat.loc[c, c] = own[c]
    print(f"\nPR AUC matrix (rows = trained on, columns = tested on; diagonal = own-country CV) -- {s}")
    display(mat.loc[COUNTRIES, COUNTRIES].round(3))

# -----------------------------------------------------------------------------------------
# Cutoffs and few-shot adaptation, on the harmonized population H1
# -----------------------------------------------------------------------------------------
ADAPT_DESIGN = "H1 drop IT size-band firms"
X, y, iso, keep = design(ADAPT_DESIGN)
cut_rows, few_rows = [], []
for a in COUNTRIES:
    src = keep & (iso == a)
    m_src = xgb(pos_w(y[src])).fit(X[src], y[src])
    p_src = m_src.predict_proba(X[src])[:, 1]
    t_src = bacc_cutoff(y[src], p_src)
    flag_src = (p_src >= t_src).mean()
    for b in COUNTRIES:
        if a == b:
            continue
        idx = np.where(keep & (iso == b))[0]
        # 5% labelled target firms ("few-shot"), 95% held out for evaluation
        lab, ev = train_test_split(idx, train_size=FEW_SHOT_SHARE, stratify=y[idx], random_state=SEED)
        p_lab, p_ev = m_src.predict_proba(X[lab])[:, 1], m_src.predict_proba(X[ev])[:, 1]
        pi_s, pi_t = y[src].mean(), y[idx].mean()   # the target failure rate is assumed known (official stats)
        rules = {
            "C0 source cutoff": t_src,
            "C1 same flag share as source": np.quantile(p_ev, 1 - flag_src),
            "C2 flag share x (target/source failure rate)": np.quantile(p_ev, 1 - min(flag_src * pi_t / pi_s, 0.99)),
            "C3 cutoff tuned on 5% labelled target": bacc_cutoff(y[lab], p_lab),
            "Oracle (best cutoff on target)": bacc_cutoff(y[ev], p_ev),
        }
        for rule, t in rules.items():
            cut_rows.append({"train": a, "test": b, "rule": rule, "flag_share": (p_ev >= t).mean(),
                             "fail_rate": y[ev].mean(), "BACC": balanced_accuracy_score(y[ev], p_ev >= t),
                             "recall": (p_ev[y[ev] == 1] >= t).mean()})
        # few-shot model variants, all evaluated on the same 95%
        pw_t = pos_w(y[lab])
        variants = {"source model only": p_ev,
                    "target-only model (5%)": xgb(pw_t, 300).fit(X[lab], y[lab]).predict_proba(X[ev])[:, 1]}
        Xp, yp = np.vstack([X[src], X[lab]]), np.r_[y[src], y[lab]]
        wp = np.r_[np.ones(src.sum()), np.full(len(lab), src.sum() / len(lab))]  # 5% sample gets equal total weight
        variants["pooled source + 5% target (up-weighted)"] = (xgb(pos_w(yp)).fit(Xp, yp, sample_weight=wp)
                                                               .predict_proba(X[ev])[:, 1])
        warm = xgb(pw_t, 150)  # continue boosting the source model on the target sample
        warm.fit(X[lab], y[lab], xgb_model=m_src.get_booster())
        variants["warm start: source + 150 trees on 5% target"] = warm.predict_proba(X[ev])[:, 1]
        for v, p in variants.items():
            few_rows.append({"train": a, "test": b, "variant": v, "n_labelled": len(lab),
                             "failures_labelled": int(y[lab].sum()),
                             "ROC_AUC": roc_auc_score(y[ev], p), "PR_AUC": average_precision_score(y[ev], p)})
    print(f"cutoffs / few-shot, source {a} done", flush=True)

cutoffs = pd.DataFrame(cut_rows)
few_shot = pd.DataFrame(few_rows)
display(cutoffs.groupby("rule", sort=False)[["BACC", "flag_share", "recall"]].mean().round(3))
display(few_shot.groupby("variant", sort=False)[["PR_AUC", "ROC_AUC"]].mean().round(3))
display(few_shot.pivot_table(index=["train", "test"], columns="variant", values="PR_AUC").round(3))

# -----------------------------------------------------------------------------------------
# Label budget: how many labelled target firms are needed? (warm start vs target-only)
# -----------------------------------------------------------------------------------------
budget_rows = []
for a in COUNTRIES:
    src = keep & (iso == a)
    m_src = xgb(pos_w(y[src])).fit(X[src], y[src])
    for b in COUNTRIES:
        if a == b:
            continue
        idx = np.where(keep & (iso == b))[0]
        pool, ev = train_test_split(idx, train_size=0.20, stratify=y[idx], random_state=SEED)  # fixed 80% eval set
        for share in (0.01, 0.02, 0.05, 0.10, 0.20):
            lab = pool if share == 0.20 else train_test_split(pool, train_size=share / 0.20, stratify=y[pool],
                                                               random_state=SEED)[0]
            warm = xgb(pos_w(y[lab]), 150)
            warm.fit(X[lab], y[lab], xgb_model=m_src.get_booster())
            own = xgb(pos_w(y[lab]), 300).fit(X[lab], y[lab])
            for v, mdl in (("warm start", warm), ("target-only", own)):
                budget_rows.append({"train": a, "test": b, "label_share": share, "n_labelled": len(lab),
                                    "failures_labelled": int(y[lab].sum()), "variant": v,
                                    "PR_AUC": average_precision_score(y[ev], mdl.predict_proba(X[ev])[:, 1])})
            if share == 0.01:
                budget_rows.append({"train": a, "test": b, "label_share": 0.0, "n_labelled": 0, "failures_labelled": 0,
                                    "variant": "warm start",
                                    "PR_AUC": average_precision_score(y[ev], m_src.predict_proba(X[ev])[:, 1])})
    print(f"label budget, source {a} done", flush=True)
budget = pd.DataFrame(budget_rows)
display(budget.pivot_table(index="label_share", columns="variant", values="PR_AUC").round(3))

transfer.to_csv(OUTPUT_DIR / "harmonization_transfer.csv", index=False)
budget.to_csv(OUTPUT_DIR / "harmonization_label_budget.csv", index=False)
within.to_csv(OUTPUT_DIR / "harmonization_within.csv", index=False)
summary.to_csv(OUTPUT_DIR / "harmonization_summary.csv", index=False)
cutoffs.to_csv(OUTPUT_DIR / "harmonization_cutoffs.csv", index=False)
few_shot.to_csv(OUTPUT_DIR / "harmonization_few_shot.csv", index=False)
