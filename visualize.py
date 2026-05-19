import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import xgboost as xgb
import shap

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)

from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# CONFIG
# ============================================================

OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("dark_background")
sns.set_context("talk")

# ============================================================
# HELPER
# ============================================================

def savefig(name):

    path = os.path.join(OUT_DIR, name)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {path}")

# ============================================================
# XGBOOST VISUALS
# ============================================================

def visualize_xgboost(model_path, feature_path):

    print("\nLoading XGBoost model...")

    model = xgb.Booster()
    model.load_model(model_path)

    print("Loading features...")

    df = pd.read_parquet(feature_path)

    feat_cols = [
        c for c in df.columns
        if c.startswith("f_")
    ]

    print(f"Detected {len(feat_cols)} feature columns")

    if len(feat_cols) == 0:
        raise ValueError("No f_* feature columns found!")

    X = df[feat_cols].values
    y = df["label"].values

    dmat = xgb.DMatrix(
        X,
        feature_names=feat_cols
    )

    probs = model.predict(dmat)

    preds = (probs >= 0.5).astype(int)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("Generating feature importance...")

    scores = model.get_score(
        importance_type="gain"
    )

    items = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:30]

    names = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(10, 8))

    plt.barh(
        names[::-1],
        vals[::-1]
    )

    plt.title("Top XGBoost Feature Importance")
    plt.xlabel("Gain")

    savefig("xgb_feature_importance.png")

    # ========================================================
    # ROC
    # ========================================================

    print("Generating ROC curve...")

    fpr, tpr, _ = roc_curve(y, probs)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))

    plt.plot(
        fpr,
        tpr,
        lw=3,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    savefig("xgb_roc.png")

    # ========================================================
    # PR CURVE
    # ========================================================

    print("Generating PR curve...")

    precision, recall, _ = precision_recall_curve(
        y,
        probs
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        recall,
        precision,
        lw=3
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    savefig("xgb_pr_curve.png")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("Generating confusion matrix...")

    cm = confusion_matrix(
        y,
        preds
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    savefig("xgb_confusion_matrix.png")

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    print("Generating probability distributions...")

    plt.figure(figsize=(9, 5))

    sns.histplot(
        probs[y == 0],
        label="Benign",
        kde=True,
        stat="density"
    )

    sns.histplot(
        probs[y == 1],
        label="Pathogenic",
        kde=True,
        stat="density"
    )

    plt.legend()

    plt.title("Prediction Probability Distribution")

    savefig("xgb_probability_distribution.png")

    # ========================================================
    # CALIBRATION CURVE
    # ========================================================

    print("Generating calibration curve...")

    prob_true, prob_pred = calibration_curve(
        y,
        probs,
        n_bins=10
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        prob_pred,
        prob_true,
        marker="o"
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")

    plt.title("Calibration Curve")

    savefig("xgb_calibration_curve.png")

    # ========================================================
    # SHAP
    # ========================================================

    print("Generating SHAP summary...")

    try:

        explainer = shap.TreeExplainer(model)

        sample_X = X[:1000]

        shap_values = explainer.shap_values(sample_X)

        shap.summary_plot(
            shap_values,
            sample_X,
            feature_names=feat_cols,
            show=False
        )

        savefig("xgb_shap_summary.png")

    except Exception as e:

        print(f"SHAP failed: {e}")

# ============================================================
# DNABERT VISUALS
# ============================================================

def visualize_dnabert(pt_path):

    print("\nLoading DNABERT checkpoint...")

    ckpt = torch.load(
        pt_path,
        map_location="cpu"
    )

    if (
        isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
    ):
        state = ckpt["model_state_dict"]

    else:
        state = ckpt

    print(f"Loaded {len(state)} tensors")

    # ========================================================
    # WEIGHT DISTRIBUTION
    # ========================================================

    print("Generating weight distributions...")

    all_weights = []

    for k, v in state.items():

        if torch.is_tensor(v):

            arr = (
                v.detach()
                .cpu()
                .numpy()
                .flatten()
            )

            if arr.size > 0:
                all_weights.append(arr)

    all_weights = np.concatenate(all_weights)

    plt.figure(figsize=(9, 5))

    plt.hist(
        all_weights,
        bins=100
    )

    plt.title("DNABERT Weight Distribution")

    plt.xlabel("Weight")
    plt.ylabel("Frequency")

    savefig("dnabert_weight_distribution.png")

    # ========================================================
    # LAYER STD
    # ========================================================

    print("Generating layer std plots...")

    names = []
    stds = []

    for k, v in state.items():

        if torch.is_tensor(v):

            arr = (
                v.detach()
                .cpu()
                .numpy()
            )

            if arr.size > 1:

                names.append(k[:40])
                stds.append(arr.std())

    top_idx = np.argsort(stds)[-30:]

    plt.figure(figsize=(12, 8))

    plt.barh(
        [names[i] for i in top_idx],
        [stds[i] for i in top_idx]
    )

    plt.title("Layer Weight Std Dev")

    savefig("dnabert_layer_std.png")

    # ========================================================
    # EMBEDDING PCA + TSNE
    # ========================================================

    print("Searching embedding layers...")

    embed_key = None

    for k in state.keys():

        if (
            "embedding" in k.lower()
            and len(state[k].shape) == 2
        ):

            embed_key = k
            break

    if embed_key:

        print(f"Using embedding layer: {embed_key}")

        emb = (
            state[embed_key]
            .detach()
            .cpu()
            .numpy()
        )

        sample = emb[:1000]

        # PCA

        print("Generating PCA...")

        pca = PCA(n_components=2)

        emb_pca = pca.fit_transform(sample)

        plt.figure(figsize=(8, 8))

        plt.scatter(
            emb_pca[:, 0],
            emb_pca[:, 1],
            s=5
        )

        plt.title("Embedding PCA")

        savefig("dnabert_embedding_pca.png")

        # TSNE

        print("Generating t-SNE...")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42
        )

        emb_tsne = tsne.fit_transform(sample)

        plt.figure(figsize=(8, 8))

        plt.scatter(
            emb_tsne[:, 0],
            emb_tsne[:, 1],
            s=5
        )

        plt.title("Embedding t-SNE")

        savefig("dnabert_embedding_tsne.png")

    # ========================================================
    # ATTENTION NORMS
    # ========================================================

    print("Generating attention norms...")

    attn_layers = []

    for k, v in state.items():

        if (
            "attention" in k.lower()
            and torch.is_tensor(v)
        ):

            attn_layers.append(
                (
                    k,
                    v.detach().cpu().numpy()
                )
            )

    norms = []
    labels = []

    for k, arr in attn_layers[:30]:

        norms.append(np.linalg.norm(arr))
        labels.append(k[:30])

    if norms:

        plt.figure(figsize=(12, 8))

        plt.barh(
            labels[::-1],
            norms[::-1]
        )

        plt.title("Attention Weight Norms")

        savefig("dnabert_attention_norms.png")

    # ========================================================
    # PARAM COUNTS
    # ========================================================

    print("Generating parameter count plot...")

    layer_sizes = {}

    for k, v in state.items():

        if torch.is_tensor(v):

            layer = k.split(".")[0]

            layer_sizes[layer] = (
                layer_sizes.get(layer, 0)
                + v.numel()
            )

    plt.figure(figsize=(10, 6))

    plt.bar(
        layer_sizes.keys(),
        layer_sizes.values()
    )

    plt.xticks(rotation=45)

    plt.title("Parameter Count By Layer")

    plt.ylabel("Parameters")

    savefig("dnabert_parameter_counts.png")

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xgb_model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dnabert_model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True
    )

    args = parser.parse_args()

    visualize_xgboost(
        args.xgb_model,
        args.features
    )

    visualize_dnabert(
        args.dnabert_model
    )

    print(
        "\nALL VISUALIZATIONS SAVED -> ./visualizations"
    )

if __name__ == "__main__":
    main()