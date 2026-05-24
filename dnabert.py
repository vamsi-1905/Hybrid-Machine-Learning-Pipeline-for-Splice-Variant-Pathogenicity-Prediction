"""
dnabert.py — DNABERT-2 fine-tuning v2
Fixes for severe class imbalance (736 benign vs ~29k pathogenic):
  - Focal loss (replaces weighted CE — handles hard examples better)
  - Uncapped oversampling weight (full 40x ratio)
  - Two-layer MLP classifier head
  - Mixup augmentation in embedding space
  - Gradient accumulation (effective batch = 128)
  - Per-class probability calibration via temperature scaling
  - MCC-based threshold tuning (more reliable than macro-F1 at 40:1)
  - Label smoothing to prevent overconfident pathogenic predictions
"""

import os, json, math, sys, re
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Windows / triton patch ────────────────────────────────────────────
def patch_cache():
    base = Path.home() / ".cache" / "huggingface"
    NEW_REBUILD = '''\
    def rebuild_alibi_tensor(self, size, device=None):
        # WIN_ALIBI_REWRITE
        import math, torch
        n = self.num_attention_heads
        def get_slopes(n):
            def _p2(n):
                s = 2**(-(2**(-(math.log2(n)-3))))
                return [s * s**i for i in range(n)]
            if math.log2(n).is_integer(): return _p2(n)
            p = 2**math.floor(math.log2(n))
            return _p2(p) + get_slopes(2*p)[::2][:n-p]
        sl  = torch.tensor(get_slopes(n), dtype=torch.float32)
        pos = torch.arange(size, dtype=torch.float32)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().unsqueeze(0)
        alibi = sl.view(-1,1,1) * -rel
        self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)
'''
    for p in base.rglob("bert_layers.py"):
        text = p.read_text(encoding="utf-8")
        if "WIN_ALIBI_REWRITE" not in text:
            m = re.search(
                r'    def rebuild_alibi_tensor\(self.*?(?=\n    def |\nclass |\Z)',
                text, flags=re.DOTALL
            )
            if m:
                text = text[:m.start()] + NEW_REBUILD + text[m.end():]
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if any(s.startswith(x) for x in
                   ["import triton","from triton","import flash_attn","from flash_attn"]
                   ) and not s.startswith("#"):
                lines.append("# WIN_REMOVED: " + line)
            else:
                lines.append(line)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in base.rglob("flash_attn_triton.py"):
        p.write_text("# WIN_STUB\n", encoding="utf-8")
    for p in base.rglob("bert_padding.py"):
        text = p.read_text(encoding="utf-8")
        if "triton" in text and "WIN_REMOVED" not in text:
            lines = [
                "# WIN_REMOVED: " + l
                if "triton" in l and not l.strip().startswith("#") else l
                for l in text.splitlines()
            ]
            p.write_text("\n".join(lines), encoding="utf-8")

patch_cache()

# Patch check_imports to ignore triton/flash_attn ImportErrors
import importlib.util as _ilu
_dmu_spec = _ilu.find_spec("transformers.dynamic_module_utils")
_dmu_mod  = _ilu.module_from_spec(_dmu_spec)
_dmu_spec.loader.exec_module(_dmu_mod)
_orig_check = _dmu_mod.check_imports
def _patched_check(filename):
    try: return _orig_check(filename)
    except ImportError as e:
        if "triton" in str(e) or "flash_attn" in str(e): return []
        raise
_dmu_mod.check_imports = _patched_check
sys.modules["transformers.dynamic_module_utils"] = _dmu_mod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, average_precision_score, matthews_corrcoef
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────
PARQUET_PATH   = "splice_windows.parquet"
MODEL_NAME     = "zhihan1996/DNABERT-2-117M"
WINDOW         = 200
MAX_LEN        = 128
BATCH_SIZE     = 32          # physical batch
ACCUM_STEPS    = 4           # effective batch = 128
EPOCHS         = 20
LR             = 8e-6        # lower LR — gentler fine-tuning on small minority
WARMUP_RATIO   = 0.15        # longer warmup
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
FOCAL_GAMMA    = 3.0         # focal loss gamma — higher = more focus on hard examples
FOCAL_ALPHA    = 0.85        # weight on benign (minority) class in focal loss
LABEL_SMOOTH   = 0.05        # prevents overconfident pathogenic predictions
MIXUP_ALPHA    = 0.3         # mixup interpolation strength (0 = off)
MIXUP_PROB     = 0.4         # probability of applying mixup per batch
TEMP_SCALE_LR  = 0.05        # temperature scaling calibration LR
SAVE_PATH      = "dnabert2_splice.pt"
RESULTS_PATH   = "dnabert2_results.json"
THRESH_PATH    = "dnabert2_threshold.json"
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE} | AMP: {USE_AMP}")


# ── Alibi patch ───────────────────────────────────────────────────────
def _safe_alibi(self, size, device=None):
    n = self.num_attention_heads
    def gs(n):
        def p2(n):
            s = 2**(-(2**(-(math.log2(n)-3))))
            return [s*s**i for i in range(n)]
        if math.log2(n).is_integer(): return p2(n)
        p = 2**math.floor(math.log2(n))
        return p2(p) + gs(2*p)[::2][:n-p]
    sl  = torch.tensor(gs(n), dtype=torch.float32)
    pos = torch.arange(size, dtype=torch.float32)
    rel = (pos.unsqueeze(0)-pos.unsqueeze(1)).abs().unsqueeze(0)
    alibi = sl.view(-1,1,1)*-rel
    self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)

def _patch_all():
    from transformers.models.bert.configuration_bert import BertConfig as StdCfg
    cache = Path.home()/".cache"/"huggingface"/"modules"
    for cfg_file in cache.rglob("configuration_bert.py"):
        mname = ".".join(cfg_file.with_suffix("").relative_to(cache).parts)
        if mname not in sys.modules:
            sp = _ilu.spec_from_file_location(mname, cfg_file)
            m  = _ilu.module_from_spec(sp); sp.loader.exec_module(m)
            m.BertConfig = StdCfg; sys.modules[mname] = m
        else:
            sys.modules[mname].BertConfig = StdCfg
    for mn, mod in list(sys.modules.items()):
        if "transformers_modules" in mn and hasattr(mod, "BertConfig"):
            mod.BertConfig = StdCfg
        if "bert_layers" in mn and hasattr(mod, "BertEncoder"):
            mod.BertEncoder.rebuild_alibi_tensor = _safe_alibi


# ── Focal Loss ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary focal loss for 2-class setting.
    alpha: weight for the POSITIVE class (label=0, benign/minority).
    gamma: focusing parameter — higher = more penalty on easy examples.
    label_smoothing: prevents the model collapsing to all-pathogenic.
    """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH):
        super().__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.smooth  = label_smoothing

    def forward(self, logits, targets):
        # logits: (B, 2), targets: (B,) with 0=benign, 1=pathogenic
        probs = torch.softmax(logits, dim=-1)           # (B, 2)
        # one-hot with label smoothing
        B = targets.size(0)
        smooth_targets = torch.full_like(probs, self.smooth / 2)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth / 2)

        # p_t = probability of the true class
        p_t = (probs * smooth_targets).sum(dim=-1)      # (B,)

        # alpha weighting: alpha for benign (0), (1-alpha) for pathogenic (1)
        alpha_t = torch.where(targets == 0,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1.0 - self.alpha, device=logits.device))

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t.clamp(min=1e-8))
        return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────
class SpliceDataset(Dataset):
    def __init__(self, df, tokenizer, window, max_len):
        self.labels = df["label"].values.astype(np.int64)
        rc, ac = f"ref_seq_{window}", f"alt_seq_{window}"
        seqs = []
        for _, row in df.iterrows():
            ref = str(row[rc]).upper() if pd.notna(row.get(rc)) else "N" * window
            alt = str(row[ac]).upper() if pd.notna(row.get(ac)) else ref
            seqs.append(ref + " [SEP] " + alt)
        print(f"  Tokenising {len(seqs):,}...")
        self.enc = tokenizer(
            seqs, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         torch.tensor(self.labels[i], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────
def load_model(device):
    _patch_all()
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()
    base = None
    for cand in ["bert_layers.BertModel", "modeling_bert.BertModel"]:
        try:
            cls = get_class_from_dynamic_module(cand, MODEL_NAME)
            from transformers.models.bert.configuration_bert import BertConfig as StdCfg
            cls.config_class = StdCfg
            base = cls.from_pretrained(
                MODEL_NAME, config=config,
                trust_remote_code=True, low_cpu_mem_usage=False
            )
            print(f"Loaded base via {cand}")
            break
        except Exception as e:
            print(f"  {cand} failed: {e}")
    if base is None:
        raise RuntimeError("Cannot load BertModel")
    _patch_all()

    class DNABERTClassifier(nn.Module):
        """
        Two-layer MLP head on top of DNABERT-2 pooled output.
        The extra hidden layer gives the model more capacity to separate
        the imbalanced classes without just memorising pathogenic patterns.
        """
        def __init__(self, b):
            super().__init__()
            self.base = b
            H = b.config.hidden_size   # 768 for DNABERT-2-117M
            self.head = nn.Sequential(
                nn.Linear(H, H // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(H // 2, H // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(H // 4, 2),
            )

        def forward(self, input_ids, attention_mask, return_embeddings=False):
            out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = (
                out[1]
                if (len(out) >= 2 and out[1] is not None and out[1].dim() == 2)
                else out[0][:, 0]
            )
            if return_embeddings:
                return pooled
            return self.head(pooled)

        def forward_from_embeddings(self, embeddings):
            return self.head(embeddings)

    return DNABERTClassifier(base).float().to(device)


# ── Mixup in embedding space ──────────────────────────────────────────
def mixup_embeddings(model, batch, alpha=MIXUP_ALPHA, device=DEVICE):
    """
    Interpolates pairs of embeddings and their labels.
    Operates in embedding space so we don't need to touch raw sequences.
    Significantly helps minority class by generating virtual benign examples.
    """
    ids  = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    lbl  = batch["labels"].to(device)

    with torch.no_grad():
        emb = model(ids, mask, return_embeddings=True)  # (B, H)

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(emb.size(0), device=device)

    mixed_emb = lam * emb + (1 - lam) * emb[idx]
    lbl_a, lbl_b = lbl, lbl[idx]
    return mixed_emb, lbl_a, lbl_b, lam


def mixup_loss(criterion, logits, lbl_a, lbl_b, lam):
    return lam * criterion(logits, lbl_a) + (1 - lam) * criterion(logits, lbl_b)


# ── Temperature scaling ───────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration. Learned on val set after training.
    Softens overconfident pathogenic probabilities.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature(model, val_loader, device):
    scaler    = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=TEMP_SCALE_LR,
                                   max_iter=100)
    ce = nn.CrossEntropyLoss()

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])

    logits_cat = torch.cat(all_logits).to(device)
    labels_cat = torch.cat(all_labels).to(device)

    def closure():
        optimizer.zero_grad()
        loss = ce(scaler(logits_cat), labels_cat)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = scaler.temperature.item()
    print(f"  Temperature: {T:.4f}  (>1 = was overconfident, <1 = was underconfident)")
    return scaler


# ── MCC threshold tuning ──────────────────────────────────────────────
def tune_threshold_mcc(probs, labels):
    """
    Optimises on MCC — much more reliable than macro-F1 at extreme imbalance.
    MCC = 0 means random, MCC = 1 means perfect.
    """
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.02, 0.98, 0.01):
        preds = (probs >= t).astype(int)
        mcc   = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t   = float(t)
    return best_t, best_mcc


# ── Training loop ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler_amp, criterion,
                accum_steps=ACCUM_STEPS, mixup_prob=MIXUP_PROB, device=DEVICE):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)

        use_mixup = (np.random.rand() < mixup_prob)

        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                if use_mixup:
                    # forward pass for embeddings (no grad needed for base in this step)
                    with torch.no_grad():
                        emb = model(ids, mask, return_embeddings=True)
                    lam    = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    idx    = torch.randperm(emb.size(0), device=device)
                    mixed  = lam * emb + (1 - lam) * emb[idx]
                    logits = model.forward_from_embeddings(mixed)
                    loss   = mixup_loss(criterion, logits, lbl, lbl[idx], lam)
                else:
                    logits = model(ids, mask)
                    loss   = criterion(logits, lbl)
                loss = loss / accum_steps
            scaler_amp.scale(loss).backward()
        else:
            if use_mixup:
                with torch.no_grad():
                    emb = model(ids, mask, return_embeddings=True)
                lam    = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx    = torch.randperm(emb.size(0), device=device)
                mixed  = lam * emb + (1 - lam) * emb[idx]
                logits = model.forward_from_embeddings(mixed)
                loss   = mixup_loss(criterion, logits, lbl, lbl[idx], lam)
            else:
                logits = model(ids, mask)
                loss   = criterion(logits, lbl)
            loss = loss / accum_steps
            loss.backward()

        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            if USE_AMP:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5, temp_scaler=None, device=DEVICE):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(ids, mask)
        else:
            logits = model(ids, mask)
        if temp_scaler is not None:
            logits = temp_scaler(logits)
        probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        all_p.extend(probs)
        all_l.extend(batch["labels"].numpy())

    p, l = np.array(all_p), np.array(all_l)
    pr   = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(l, pr),
        "auc":      roc_auc_score(l, p),
        "pr_auc":   average_precision_score(l, p),
        "f1_path":  f1_score(l, pr, pos_label=1, zero_division=0),
        "f1_ben":   f1_score(l, pr, pos_label=0, zero_division=0),
        "mcc":      matthews_corrcoef(l, pr),
        "probs":    p,
        "labels":   l,
    }


# ── Main ──────────────────────────────────────────────────────────────
def run():
    df = pd.read_parquet(PARQUET_PATH)
    window = WINDOW
    if f"ref_seq_{window}" not in df.columns:
        window = int(
            [c for c in df.columns if c.startswith("ref_seq_")][0].split("_")[-1]
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    n_ben = (train_df["label"] == 0).sum()
    n_pat = (train_df["label"] == 1).sum()
    ratio = n_pat / max(n_ben, 1)
    print(f"\nBenign={n_ben:,}  Pathogenic={n_pat:,}  ratio={ratio:.1f}x")

    # ── Focal loss (replaces weighted CE) ──
    # alpha=0.85 means 85% weight on benign class in focal loss
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    print(f"Focal loss: alpha={FOCAL_ALPHA} (benign weight) gamma={FOCAL_GAMMA} smooth={LABEL_SMOOTH}")

    # ── Oversampling: FULL ratio, no cap ──
    # Previous code capped at 15x — for 40:1 imbalance this wasn't enough
    sample_weights = np.where(train_df["label"].values == 0, float(ratio), 1.0)
    n_samples      = int(2 * n_pat)   # each epoch sees ~2x pathogenic count total
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = n_samples,
        replacement = True,
    )
    print(f"Oversampling: {n_samples:,} samples/epoch (benign weight={ratio:.1f}x, uncapped)")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()

    print("Loading model...")
    model = load_model(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    head_params  = sum(p.numel() for p in model.head.parameters())
    print(f"Total params: {total_params:,}  Head params: {head_params:,}")

    print("\nBuilding datasets...")
    train_ds = SpliceDataset(train_df, tokenizer, window, MAX_LEN)
    val_ds   = SpliceDataset(val_df,   tokenizer, window, MAX_LEN)
    test_ds  = SpliceDataset(test_df,  tokenizer, window, MAX_LEN)

    kw = dict(num_workers=0, pin_memory=USE_AMP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False, **kw)

    # ── Differential LR: smaller LR for base, higher for head ──
    no_decay = ["bias", "LayerNorm.weight"]
    base_params = [
        {"params": [p for n, p in model.base.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY, "lr": LR},
        {"params": [p for n, p in model.base.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": LR},
    ]
    head_params_groups = [
        {"params": [p for n, p in model.head.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY, "lr": LR * 10},   # 10x LR for head
        {"params": [p for n, p in model.head.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": LR * 10},
    ]
    optimizer = torch.optim.AdamW(base_params + head_params_groups)

    total_steps = EPOCHS * (len(train_loader) // ACCUM_STEPS)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(WARMUP_RATIO * total_steps),
        num_training_steps= total_steps,
    )
    scaler_amp = torch.amp.GradScaler(device="cuda") if USE_AMP else None

    print(f"\nTraining {EPOCHS} epochs | effective batch={BATCH_SIZE*ACCUM_STEPS} | "
          f"mixup_prob={MIXUP_PROB}\n")

    best_auc = 0.0
    losses, history = [], []

    for epoch in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_loader, optimizer, scheduler, scaler_amp, criterion)
        vm = evaluate(model, val_loader, threshold=0.5)   # use 0.5 during training
        losses.append(tl)
        history.append(vm)
        print(
            f"Ep {epoch:02d}/{EPOCHS}  loss={tl:.4f}  auc={vm['auc']:.4f}  "
            f"pr_auc={vm['pr_auc']:.4f}  f1_ben={vm['f1_ben']:.4f}  "
            f"f1_path={vm['f1_path']:.4f}  mcc={vm['mcc']:.4f}"
        )
        if vm["auc"] > best_auc:
            best_auc = vm["auc"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ saved (AUC={best_auc:.4f})")

    # ── Load best checkpoint ──
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))

    # ── Temperature calibration on val set ──
    print("\nCalibrating temperature...")
    temp_scaler = calibrate_temperature(model, val_loader, DEVICE)

    # ── MCC-based threshold tuning ──
    vm_best = evaluate(model, val_loader, threshold=0.5, temp_scaler=temp_scaler)
    best_t, best_mcc = tune_threshold_mcc(vm_best["probs"], vm_best["labels"])
    macro_f1 = (
        f1_score(vm_best["labels"], (vm_best["probs"] >= best_t).astype(int),
                 pos_label=0, zero_division=0) +
        f1_score(vm_best["labels"], (vm_best["probs"] >= best_t).astype(int),
                 pos_label=1, zero_division=0)
    ) / 2
    print(f"MCC threshold: {best_t:.2f}  MCC={best_mcc:.4f}  macro-F1={macro_f1:.4f}")

    json.dump(
        {"threshold": best_t, "val_mcc": best_mcc, "val_macro_f1": macro_f1,
         "temperature": temp_scaler.temperature.item()},
        open(THRESH_PATH, "w"), indent=2
    )

    # ── Test evaluation ──
    tm = evaluate(model, test_loader, threshold=best_t, temp_scaler=temp_scaler)
    print(f"\n── Test Results ──")
    print(f"AUC={tm['auc']:.4f}  PR-AUC={tm['pr_auc']:.4f}  MCC={tm['mcc']:.4f}")
    print(classification_report(
        tm["labels"], (tm["probs"] >= best_t).astype(int),
        target_names=["Benign", "Pathogenic"]
    ))

    json.dump({
        "test_auc":       tm["auc"],
        "test_pr_auc":    tm["pr_auc"],
        "test_mcc":       tm["mcc"],
        "test_f1_benign": tm["f1_ben"],
        "test_f1_path":   tm["f1_path"],
        "best_val_auc":   best_auc,
        "threshold":      best_t,
        "temperature":    temp_scaler.temperature.item(),
        "model":          MODEL_NAME,
        "window":         window,
        "epochs":         EPOCHS,
        "focal_gamma":    FOCAL_GAMMA,
        "focal_alpha":    FOCAL_ALPHA,
    }, open(RESULTS_PATH, "w"), indent=2)

    # ── Training curves ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    ep = range(1, len(losses) + 1)
    for ax, vals, title in zip(axes, [
        losses,
        [m["auc"]     for m in history],
        [m["pr_auc"]  for m in history],
        [m["f1_ben"]  for m in history],
        [m["mcc"]     for m in history],
    ], ["Train Loss", "Val AUC", "Val PR-AUC", "Val F1 Benign", "Val MCC"]):
        ax.plot(ep, vals, marker="o", markersize=3)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dnabert2_curves.png", dpi=150)
    plt.close()
    print("\nDone. Curves saved → dnabert2_curves.png")


if __name__ == "__main__":
    run()