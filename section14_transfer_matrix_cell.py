# =============================================================================
# 14. Train on one country, predict another: the 4x4 transfer matrix
# Paste as ONE code cell after section 13. Self-contained: needs LAGGED_FILE, OUTPUT_DIR.
# Runtime: ~3-5 min on Colab (12 cross-country fits + 4 within-country 5-fold CVs).
# Diagonal = within-country 5-fold out-of-fold performance; off-diagonal = model trained on the
# row country (all its firms) and applied unchanged to the column country.
# 'cutoff' columns: the balanced-accuracy cutoff chosen on the training country, applied as is.
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TM_COUNTRIES = ["IT", "FR", "PT", "ES"]
TM_NUM = ["Number_of_patents", "Number_of_trademarks", "consdummy", "tfp_acf", "fin_rev", "int_paid", "ebitda",
          "cash_flow", "depr", "revenue", "total_assets", "long_term_debt", "employees", "added_value", "materials",
          "wage_bill", "loans", "int_fixed_assets", "fixed_assets", "current_liabilities", "liquidity_ratio",
          "solvency_ratio", "current_assets", "fin_expenses", "net_income", "fin_cons100", "inv", "real_SA",
          "shareholders_funds", "NEG_VA", "ICR_failure", "profitability", "misallocated_fixed", "interest_diff"]
DROP_IT_BANDED = False  # True = remove IT firms that only report a revenue size band (see section 5b)

tm = pd.read_csv(LAGGED_FILE, usecols=["iso", "failure", "nace_2", "control"] + TM_NUM, low_memory=False)
tm = tm.dropna(subset=["failure"]).reset_index(drop=True)
if DROP_IT_BANDED:
    tm = tm[~(tm["iso"].eq("IT") & tm["total_assets"].isna()
              & tm["revenue"].isin([250_000, 750_000, 1_500_000, 3_500_000]))].reset_index(drop=True)
X_tm = np.hstack([tm[TM_NUM].to_numpy(float),
                  pd.get_dummies(tm[["nace_2", "control"]].astype("string").fillna("__MISSING__"), dtype=float).to_numpy()])
y_tm, iso_tm = tm["failure"].astype(int).to_numpy(), tm["iso"].to_numpy()


def tm_xgb(y_train):
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=5, reg_lambda=2.0, eval_metric="logloss", tree_method="hist",
                         scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42, n_jobs=-1)


def tm_cutoff(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, p >= t) for t in grid]))])


tm_rows = []
for a in TM_COUNTRIES:
    src = iso_tm == a
    # diagonal: within-country out-of-fold
    p_oof = np.zeros(src.sum())
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X_tm[src], y_tm[src]):
        p_oof[te] = tm_xgb(y_tm[src][tr]).fit(X_tm[src][tr], y_tm[src][tr]).predict_proba(X_tm[src][te])[:, 1]
    tm_rows.append({"train": a, "test": a, "ROC_AUC": roc_auc_score(y_tm[src], p_oof),
                    "PR_AUC": average_precision_score(y_tm[src], p_oof), "fail_rate": y_tm[src].mean()})
    model = tm_xgb(y_tm[src]).fit(X_tm[src], y_tm[src])
    cut = tm_cutoff(y_tm[src], model.predict_proba(X_tm[src])[:, 1])
    for b in TM_COUNTRIES:
        if b == a:
            continue
        tgt = iso_tm == b
        p = model.predict_proba(X_tm[tgt])[:, 1]
        tm_rows.append({"train": a, "test": b, "ROC_AUC": roc_auc_score(y_tm[tgt], p),
                        "PR_AUC": average_precision_score(y_tm[tgt], p), "fail_rate": y_tm[tgt].mean(),
                        "share_flagged_at_train_cutoff": (p >= cut).mean(),
                        "BACC_at_train_cutoff": balanced_accuracy_score(y_tm[tgt], p >= cut)})
    print("trained on", a, "done")

transfer_matrix = pd.DataFrame(tm_rows)
for metric in ["PR_AUC", "ROC_AUC", "share_flagged_at_train_cutoff"]:
    print(f"\n{metric} (rows = trained on, columns = tested on)")
    display(transfer_matrix.pivot(index="train", columns="test", values=metric)
            .loc[TM_COUNTRIES, TM_COUNTRIES].round(3))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(transfer_matrix.pivot(index="train", columns="test", values="PR_AUC").loc[TM_COUNTRIES, TM_COUNTRIES],
            annot=True, fmt=".3f", cmap="Purples", vmin=0.4, vmax=0.8, ax=ax)
ax.set(title="PR AUC: trained on row, tested on column", xlabel="Tested on", ylabel="Trained on")
plt.tight_layout()
plt.show()
transfer_matrix.to_csv(OUTPUT_DIR / "transfer_matrix.csv", index=False)
