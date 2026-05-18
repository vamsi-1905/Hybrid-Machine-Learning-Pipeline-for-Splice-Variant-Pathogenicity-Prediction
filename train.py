"""
train.py — XGBoost with correct imbalance handling
label=0 = BENIGN (minority ~736), label=1 = PATHOGENIC (majority ~29k)
"""
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
    roc_curve, average_precision_score, precision_recall_curve
)

IN_PATH     = "features.parquet"
MODEL_OUT   = "xgb_model.json"
RESULTS_OUT = "xgb_results.json"
THRESH_OUT  = "xgb_threshold.json"
EARLY_STOP  = 50
N_EST       = 1000
MAX_DEPTH   = 6
LR          = 0.05
SUBSAMPLE   = 0.8
COLSAMPLE   = 0.8


def load_splits(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"  Rows={len(df):,}  Features={len(feat_cols)}")
    for sp in ["train","val","test"]:
        s = df[df["split"]==sp]
        n0,n1 = (s["label"]==0).sum(),(s["label"]==1).sum()
        print(f"  {sp}: benign={n0:,}  pathogenic={n1:,}  ratio={n1/max(n0,1):.1f}:1")
    tr = df[df["split"]=="train"]
    va = df[df["split"]=="val"]
    te = df[df["split"]=="test"]
    def xy(d): return d[feat_cols].values, d["label"].values
    return *xy(tr), *xy(va), *xy(te), feat_cols


def tune_threshold(probs, labels):
    best_t, best = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.02):
        p = (probs>=t).astype(int)
        f = (f1_score(labels,p,pos_label=0,zero_division=0)+
             f1_score(labels,p,pos_label=1,zero_division=0))/2
        if f>best: best,best_t=f,float(t)
    return best_t, best


def run():
    X_tr,y_tr, X_va,y_va, X_te,y_te, feat_cols = load_splits(IN_PATH)

    n_ben = (y_tr==0).sum()
    n_pat = (y_tr==1).sum()
    # scale_pos_weight: XGBoost binary:logistic treats label=1 as positive.
    # label=1=pathogenic is the MAJORITY here, so spw < 1 to penalise
    # over-predicting pathogenic and force the model to also learn benign.
    spw = 1.0
    print(f"\nBenign={n_ben:,}  Pathogenic={n_pat:,}  scale_pos_weight={spw:.4f}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feat_cols)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : ["logloss","auc","aucpr"],
        "max_depth"        : MAX_DEPTH,
        "learning_rate"    : LR,
        "subsample"        : SUBSAMPLE,
        "colsample_bytree" : COLSAMPLE,
        "scale_pos_weight" : spw,
        "min_child_weight" : 3,
        "gamma"            : 0.1,
        "seed"             : 42,
        "tree_method"      : "hist",
        "device"           : "cuda" if _cuda() else "cpu",
    }

    print(f"\nTraining XGBoost (device={params['device']})...")
    ev = {}
    model = xgb.train(params, dtrain, num_boost_round=N_EST,
                      evals=[(dtrain,"train"),(dval,"val")],
                      early_stopping_rounds=EARLY_STOP,
                      evals_result=ev, verbose_eval=50)
    model.save_model(MODEL_OUT)
    print(f"Saved → {MODEL_OUT}")

    val_p = model.predict(dval)
    print(f"\nVal probs: min={val_p.min():.4f} mean={val_p.mean():.4f} max={val_p.max():.4f}")
    best_t, best_macro = tune_threshold(val_p, y_va)
    print(f"Optimal threshold: {best_t:.2f}  macro-F1={best_macro:.4f}")
    json.dump({"threshold":best_t,"val_macro_f1":best_macro}, open(THRESH_OUT,"w"), indent=2)

    print("\n── Evaluation ──")
    for name,dmat,y in [("Val",dval,y_va),("Test",dtest,y_te)]:
        p = model.predict(dmat)
        pr = (p>=best_t).astype(int)
        print(f"\n{name}: AUC={roc_auc_score(y,p):.4f}  PR-AUC={average_precision_score(y,p):.4f}")
        print(classification_report(y, pr, target_names=["Benign","Pathogenic"]))
        cm = confusion_matrix(y, pr)
        fig,ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                    xticklabels=["Benign","Pathogenic"],yticklabels=["Benign","Pathogenic"],ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion ({name})")
        plt.tight_layout(); plt.savefig(f"xgb_confusion_{name.lower()}.png",dpi=150); plt.close()
        if name=="Test":
            fpr,tpr,_ = roc_curve(y,p)
            fig,ax=plt.subplots(figsize=(5,4))
            ax.plot(fpr,tpr,lw=2,label=f"AUC={roc_auc_score(y,p):.4f}")
            ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve"); ax.legend(); plt.tight_layout()
            plt.savefig("xgb_roc.png",dpi=150); plt.close()

    # Feature importance
    scores = model.get_score(importance_type="gain")
    items  = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:30]
    names  = [feat_cols[int(k[2:])] if k.startswith("f_") else k for k,_ in items]
    fig,ax = plt.subplots(figsize=(9,7))
    ax.barh(names[::-1],[v for _,v in items[::-1]]); ax.set_xlabel("Gain")
    ax.set_title("Top 30 Feature Importances")
    plt.tight_layout(); plt.savefig("xgb_feature_importance.png",dpi=150); plt.close()

    tp = model.predict(dtest)
    pp = (tp>=best_t).astype(int)
    results = {
        "test_auc":       float(roc_auc_score(y_te,tp)),
        "test_pr_auc":    float(average_precision_score(y_te,tp)),
        "test_accuracy":  float(accuracy_score(y_te,pp)),
        "test_f1_benign": float(f1_score(y_te,pp,pos_label=0,zero_division=0)),
        "test_f1_path":   float(f1_score(y_te,pp,pos_label=1,zero_division=0)),
        "threshold":      best_t,
        "n_benign_train": int(n_ben),
        "n_path_train":   int(n_pat),
        "best_iter":      int(model.best_iteration),
    }
    json.dump(results, open(RESULTS_OUT,"w"), indent=2)
    print(f"\n{json.dumps(results,indent=2)}")


def _cuda():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"],capture_output=True).returncode==0
    except: return False


if __name__ == "__main__":
    run()