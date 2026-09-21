# =============================================================================
# 13. Homework: cross-country generalization (leave-one-country-out)
# Paste as ONE code cell after section 12. Reuses from the notebook:
#   LAGGED_FILE, NUMERIC_LAGGED, N_ESTIMATORS, RANDOM_STATE, OUTPUT_DIR,
#   dense_ohe(), best_bacc_threshold()
# Set N_ESTIMATORS = 500 / QUICK_MODE = False for the numbers you hand in.
# Runtime on a standard Colab CPU: roughly 5-8 minutes for both samples.
# =============================================================================
LOCO_COUNTRIES = ["IT", "FR", "PT", "ES"]
LOCO_FOLDS = 5
LOCO_CAT = ["nace_2", "control"]  # 'iso' deliberately dropped: it is constant within a country
                                  # and the held-out country's dummy would be unseen in training

pooled = pd.read_csv(LAGGED_FILE, usecols=list(dict.fromkeys(["id", "iso", "failure"] + NUMERIC_LAGGED + LOCO_CAT)),
                     low_memory=False).dropna(subset=["failure"])
pooled["failure"] = pooled["failure"].astype(int)
pooled["nace_2"] = pooled["nace_2"].astype("string")

# Data artifact found while doing this exercise: ~149k Italian firms have NO balance sheet and a revenue
# equal to a size-band value (250k / 750k / 1.5M / 3.5M EUR). Not one of them fails. In FR/PT/ES a missing
# balance sheet goes with HIGHER failure risk, in IT (because of these rows) with LOWER risk.
banded_it = (pooled["iso"].eq("IT") & pooled["total_assets"].isna()
             & pooled["revenue"].isin([250_000, 750_000, 1_500_000, 3_500_000]))
print("IT firms with banded revenue & no balance sheet:", int(banded_it.sum()),
      "| failure rate among them:", f"{pooled.loc[banded_it, 'failure'].mean():.2%}")
display(pooled.assign(no_balance_sheet=pooled["total_assets"].isna())
        .pivot_table(index="iso", columns="no_balance_sheet", values="failure", aggfunc="mean")
        .rename(columns={False: "fail_rate_with_balance_sheet", True: "fail_rate_no_balance_sheet"})
        .style.format("{:.2%}"))

LOCO_SAMPLES = {"as_delivered": pooled, "IT_banded_removed": pooled.loc[~banded_it].copy()}


def loco_xgb(y_train):
    pos_w = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_LAGGED),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                          ("ohe", dense_ohe())]), LOCO_CAT)])
    return Pipeline([("pre", pre), ("model", XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=pos_w, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1))])


def loco_metrics(y, p, thr_train=None):
    t = best_bacc_threshold(y, p)  # same (optimistic) rule as section 6, for comparability
    out = {"n": len(y), "fail_rate": y.mean(), "ROC_AUC": roc_auc_score(y, p),
           "PR_AUC": average_precision_score(y, p), "F1": f1_score(y, p >= t), "BACC": balanced_accuracy_score(y, p >= t)}
    out["PR_lift"] = out["PR_AUC"] / out["fail_rate"]  # PR AUC relative to a random classifier
    if thr_train is not None:  # honest version: cutoff chosen on the training countries, applied as-is
        out["BACC_train_thr"] = balanced_accuracy_score(y, p >= thr_train)
        out["share_flagged_train_thr"] = (p >= thr_train).mean()
    return out


loco_rows = []
for sample_name, S in LOCO_SAMPLES.items():
    X_all, y_all, iso_all = S[NUMERIC_LAGGED + LOCO_CAT], S["failure"].to_numpy(), S["iso"].to_numpy()
    for c in LOCO_COUNTRIES:
        # (A) within-country benchmark: train and validate on country c only (5-fold out-of-fold)
        m = iso_all == c
        cv = StratifiedKFold(n_splits=LOCO_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        p = cross_val_predict(loco_xgb(y_all[m]), X_all[m], y_all[m], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        loco_rows.append({"sample": sample_name, "design": "A: within-country", "test_country": c, **loco_metrics(y_all[m], p)})

        # (C) leave-one-country-out: train on the other three pooled, test on c (never seen)
        tr, te = iso_all != c, iso_all == c
        mdl = loco_xgb(y_all[tr]).fit(X_all[tr], y_all[tr])
        thr_train = best_bacc_threshold(y_all[tr], mdl.predict_proba(X_all[tr])[:, 1])
        p = mdl.predict_proba(X_all[te])[:, 1]
        loco_rows.append({"sample": sample_name, "design": "C: leave-one-country-out", "test_country": c,
                          **loco_metrics(y_all[te], p, thr_train)})
        print(f"{sample_name:18s} {c}: within PR AUC={loco_rows[-2]['PR_AUC']:.3f} | LOCO PR AUC={loco_rows[-1]['PR_AUC']:.3f}")

loco = pd.DataFrame(loco_rows)
loco_gap = (loco.pivot_table(index=["sample", "test_country"], columns="design", values=["ROC_AUC", "PR_AUC"])
            .pipe(lambda t: t.set_axis([f"{a} | {b[:1]}" for a, b in t.columns], axis=1)))
loco_gap["PR_AUC drop (C-A)"] = loco_gap["PR_AUC | C"] - loco_gap["PR_AUC | A"]
loco_gap["ROC_AUC drop (C-A)"] = loco_gap["ROC_AUC | C"] - loco_gap["ROC_AUC | A"]
display(loco.round(3))
display(loco_gap.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (sample_name, g) in zip(axes, loco.groupby("sample", sort=False)):
    sns.barplot(data=g, x="test_country", y="PR_AUC", hue="design", order=LOCO_COUNTRIES,
                palette=["#6A1B9A", "#B39DDB"], ax=ax)
    ax.set(title=f"PR AUC by test country ({sample_name})", xlabel="Test country", ylabel="PR AUC")
    ax.legend(fontsize=8, title="")
plt.tight_layout()
plt.show()

loco.to_csv(OUTPUT_DIR / "loco_results.csv", index=False)
