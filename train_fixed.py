

import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    average_precision_score, matthews_corrcoef,
)

IN_PATH = "features_2.parquet"
MODEL_OUT   = "xgb_model.json"
RESULTS_OUT = "xgb_results.json"
THRESH_OUT  = "xgb_threshold.json"

EARLY_STOP  = 30
N_EST       = 500
MAX_DEPTH   = 4      # was 6 — shallower = less memorisation
LR          = 0.05
SUBSAMPLE   = 0.6    # was 0.8 — see fewer rows per tree
COLSAMPLE   = 0.6    # was 0.8 — see fewer features per tree
MIN_CHILD_W = 10     # was 3 — require 10 samples per leaf
GAMMA       = 1.0    # was 0.1 — much higher pruning
REG_ALPHA   = 1.0    # was 0.1 — stronger L1
REG_LAMBDA  = 5.0    # was 1.5 — much stronger L2

# Hold out entire chromosomes — the ONLY honest split when data has duplicates
TEST_CHROMS = {"8", "21"}
VAL_CHROMS  = {"7", "X"}


def load_and_dedup(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("f_")]

    before = len(df)
    df = df.drop_duplicates(subset=["chrom","position","ref","alt"])
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"  ⚠  Removed {removed:,} duplicates (repair.py artefact) — "
              f"{after:,} unique variants remain")
    else:
        print(f"  No duplicates found — {after:,} rows")

    n0 = (df["label"]==0).sum()
    n1 = (df["label"]==1).sum()
    print(f"  After dedup: benign={n0:,}  pathogenic={n1:,}  ratio={n1/max(n0,1):.1f}:1")
    print(f"  Features: {len(feat_cols)}")
    return df, feat_cols


def make_chrom_split(df):
    chrom = df["chrom"].astype(str)
    te_mask = chrom.isin(TEST_CHROMS)
    va_mask = chrom.isin(VAL_CHROMS)
    tr_mask = ~te_mask & ~va_mask

    tr = df[tr_mask].copy()
    va = df[va_mask].copy()
    te = df[te_mask].copy()

    print("\nChromosome split (no sequence leakage):")
    for name, sub in [("Train",tr),("Val",va),("Test",te)]:
        n0 = (sub["label"]==0).sum()
        n1 = (sub["label"]==1).sum()
        chroms = sorted(sub["chrom"].astype(str).unique())
        print(f"  {name:5} {len(sub):>6,}  benign={n0:,}  path={n1:,}  "
              f"chroms={chroms[:6]}{'...' if len(chroms)>6 else ''}")

    # If val/test too small (chromosome distribution uneven), fall back
    if len(va) < 100:
        print("  ⚠  Val set too small — using random 15% of train instead")
        idx   = np.random.RandomState(42).permutation(len(tr))
        n_val = max(int(len(tr)*0.15), 500)
        va    = tr.iloc[idx[:n_val]]
        tr    = tr.iloc[idx[n_val:]]
    if len(te) < 100:
        print("  ⚠  Test set too small — using random 15% of remaining train")
        idx    = np.random.RandomState(42).permutation(len(tr))
        n_test = max(int(len(tr)*0.15), 500)
        te     = tr.iloc[idx[:n_test]]
        tr     = tr.iloc[idx[n_test:]]

    return tr, va, te


def tune_threshold(probs, labels):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        mcc   = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t   = float(t)
    return best_t, best_mcc


def _cuda():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except:
        return False


def run():
    df, feat_cols = load_and_dedup(IN_PATH)

    # Use chromosome split regardless of existing "split" column
    # (existing split was random → leakage from duplicate rows)
    tr, va, te = make_chrom_split(df)

    def xy(d): return d[feat_cols].values, d["label"].values
    X_tr,y_tr = xy(tr)
    X_va,y_va = xy(va)
    X_te,y_te = xy(te)

    n_ben = (y_tr==0).sum()
    n_pat = (y_tr==1).sum()
    spw   = n_pat / max(n_ben, 1)
    print(f"\nTrain: benign={n_ben:,}  path={n_pat:,}  spw={spw:.3f}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feat_cols)

    device = "cuda" if _cuda() else "cpu"
    params = {
        "objective":         "binary:logistic",
        "eval_metric":       ["logloss","auc","aucpr"],
        "max_depth":         MAX_DEPTH,
        "learning_rate":     LR,
        "subsample":         SUBSAMPLE,
        "colsample_bytree":  COLSAMPLE,
        "scale_pos_weight":  spw,
        "min_child_weight":  MIN_CHILD_W,
        "gamma":             GAMMA,
        "reg_alpha":         REG_ALPHA,
        "reg_lambda":        REG_LAMBDA,
        "seed":              42,
        "tree_method":       "hist",
        "device":            device,
    }

    print(f"\nTraining XGBoost (device={device})  "
          f"max_depth={MAX_DEPTH}  lambda={REG_LAMBDA}  early_stop={EARLY_STOP}...")
    ev = {}
    model = xgb.train(
        params, dtrain,
        num_boost_round       = N_EST,
        evals                 = [(dtrain,"train"),(dval,"val")],
        early_stopping_rounds = EARLY_STOP,
        evals_result          = ev,
        verbose_eval          = 25,
    )
    model.save_model(MODEL_OUT)

    best_i          = model.best_iteration
    train_auc_best  = ev["train"]["auc"][best_i]
    val_auc_best    = ev["val"]["auc"][best_i]
    gap             = train_auc_best - val_auc_best

    print(f"\nBest iter: {best_i}")
    print(f"Train AUC: {train_auc_best:.4f}   Val AUC: {val_auc_best:.4f}   Gap: {gap:.4f}")
    if gap > 0.05:
        print("⚠  STILL OVERFITTING — increase REG_LAMBDA further or reduce MAX_DEPTH to 3")
    elif gap > 0.02:
        print("⚡ Mild overfit — acceptable, watch test AUC")
    else:
        print("✓ Train-val gap healthy")

    # Threshold tuning on val
    val_p = model.predict(dval)
    print(f"\nVal probs: min={val_p.min():.4f}  mean={val_p.mean():.4f}  max={val_p.max():.4f}")
    best_t, best_mcc = tune_threshold(val_p, y_va)
    macro_f1 = (
        f1_score(y_va,(val_p>=best_t).astype(int),pos_label=0,zero_division=0) +
        f1_score(y_va,(val_p>=best_t).astype(int),pos_label=1,zero_division=0)
    ) / 2
    print(f"Threshold: {best_t:.2f}  MCC={best_mcc:.4f}  macro-F1={macro_f1:.4f}")
    json.dump({"threshold":best_t,"val_mcc":best_mcc,"val_macro_f1":macro_f1},
              open(THRESH_OUT,"w"), indent=2)

    # Evaluation
    print("\n── Evaluation ──")
    for name, dmat, y in [("Val",dval,y_va),("Test",dtest,y_te)]:
        p   = model.predict(dmat)
        pr  = (p>=best_t).astype(int)
        auc = roc_auc_score(y,p)
        mcc = matthews_corrcoef(y,pr)
        pr_auc = average_precision_score(y,p)
        print(f"\n{name}: AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  MCC={mcc:.4f}")
        print(classification_report(y,pr,target_names=["Benign","Pathogenic"]))
        cm = confusion_matrix(y,pr)
        fig,ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                    xticklabels=["Benign","Pathogenic"],
                    yticklabels=["Benign","Pathogenic"],ax=ax)
        ax.set_title(f"Confusion ({name})")
        plt.tight_layout(); plt.savefig(f"xgb_confusion_{name.lower()}.png",dpi=150); plt.close()

    # Training curves
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    rds = range(len(ev["train"]["logloss"]))
    for ax,key,title in zip(axes,["logloss","auc","aucpr"],["Log Loss","AUC","PR-AUC"]):
        ax.plot(rds, ev["train"][key], label="train", lw=2)
        ax.plot(rds, ev["val"][key],   label="val",   lw=2)
        ax.axvline(best_i, color="red", ls="--", alpha=0.5, label="best")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f"Train-Val AUC gap: {gap:.4f}", color="red" if gap>0.05 else "green")
    plt.tight_layout(); plt.savefig("xgb_curves.png",dpi=150); plt.close()

    # Feature importance
    scores = model.get_score(importance_type="gain")
    items  = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:30]
    names  = [feat_cols[int(k[2:])] if k.startswith("f_") else k for k,_ in items]
    fig,ax = plt.subplots(figsize=(9,7))
    ax.barh(names[::-1],[v for _,v in items[::-1]])
    ax.set_xlabel("Gain"); ax.set_title("Top 30 Feature Importances")
    plt.tight_layout(); plt.savefig("xgb_feature_importance.png",dpi=150); plt.close()

    # Save results
    tp = model.predict(dtest)
    pp = (tp>=best_t).astype(int)
    results = {
        "test_auc":           float(roc_auc_score(y_te,tp)),
        "test_pr_auc":        float(average_precision_score(y_te,tp)),
        "test_accuracy":      float(accuracy_score(y_te,pp)),
        "test_f1_benign":     float(f1_score(y_te,pp,pos_label=0,zero_division=0)),
        "test_f1_path":       float(f1_score(y_te,pp,pos_label=1,zero_division=0)),
        "test_mcc":           float(matthews_corrcoef(y_te,pp)),
        "train_auc_at_best":  float(train_auc_best),
        "val_auc_at_best":    float(val_auc_best),
        "train_val_gap":      float(gap),
        "threshold":          best_t,
        "best_iter":          int(best_i),
        "split":              "chromosome-based (test=chr8,21  val=chr7,X)",
    }
    json.dump(results, open(RESULTS_OUT,"w"), indent=2)
    print(f"\n── Final ──\n{json.dumps(results,indent=2)}")


if __name__ == "__main__":
    run()